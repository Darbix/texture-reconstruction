import os
import gc
import cv2
import torch
import shutil
import argparse
from itertools import islice

from image_utils import load_data_iteratively, save_img, plot_image
from align_by_uv_mapping import align_image_uv
from align_by_homography_opticalflow import init_searaft, get_ref_image,\
    align_image_hg_of, print_ram, Settings

VIEW_IMGS_DIR = "color_imgs" # Name of a directory for view images
UV_IMGS_DIR = "uv_imgs"      # Name of a directory for UV maps
DATA_INFO_FILE = "data.txt"  # Name of a file with a source texture name
DATA_INFO_KEY = "data_txt"   # A directory key for DATA_INFO_FILE


def parse_arguments():
    """Argument parser"""
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("--data_path", type=str,
        help="Path to the input dataset")
    parser.add_argument("--texture_path", type=str,
        help="Path to textures")
    parser.add_argument("--output_path", type=str,
        help="Path to the output dataset directory")
    parser.add_argument("--hgof", action='store_true',
        help="Enable HGOF homography + optical flow mode instead UV mapping")
    parser.add_argument("--iters", type=int, default=4,
        help="Number of optical flow iterations")
    parser.add_argument("--max_size", type=int, default=1500,
        help="Maximal size of the input image to the SEA-RAFT (-1 to keep original)")
    parser.add_argument("--mask", action='store_true',
        help="Mask out alpha channels using UV data (remove a background)")
    parser.add_argument("--uv_upscale", type=float, default=1.5,
        help="UV map upscale factor compared to the texture size for UV mapping")
    parser.add_argument("--range", type=parse_range, required=True, 
                        help="Specify range as 'start:stop' (e.g., 50:100)")

    return parser.parse_args()


def parse_range(range_str):
    """Parses a range string like '50:100' into (start, stop) integers."""
    try:
        if(range_str is None):
            return None
        start, stop = map(int, range_str.split(":"))
        return start, stop
    except ValueError:
        raise argparse.ArgumentTypeError("Range must be in 'start:stop' (with meaning data[start:stop]) format with integers.")


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    DATA_PATH = args.data_path
    try:
        data_range = args.range#ap(int, args.range.split(":")) # Range of loaded data
    except Exception as e:
        print(e)
        exit(1)
    start, stop = data_range if data_range is not None else (0, -1)

    TEXTURE_PATH = args.texture_path
    OUTPUT_PATH = args.output_path
    TOGGLE_UV_OR_HGOF = 'HGOF' if args.hgof else 'UV'
    OF_ITERS = args.iters
    OF_MAX_SIZE = args.max_size
    MASK_OBJECT = args.mask
    UV_UPSCALE_FACTOR = args.uv_upscale

    if None in [DATA_PATH, TEXTURE_PATH, OUTPUT_PATH]:
        raise ValueError("One or more required paths are not set.")

    # Initialization
    # Optical flow SEA-RAFT model
    settings = Settings(iterations=OF_ITERS, img_max_size=OF_MAX_SIZE)
    model, model_args = init_searaft(settings) if TOGGLE_UV_OR_HGOF == 'HGOF' else (None, None)
    
    # Load pairs of scenes and dictionaries of files
    data_iter = load_data_iteratively(DATA_PATH, VIEW_IMGS_DIR, UV_IMGS_DIR,
                                      DATA_INFO_FILE, DATA_INFO_KEY)
    try:
        if(start > stop and stop != -1 or start == stop or start < 0 or stop < -1):
            raise ValueError(f"An invalid data range given:", start, stop)
        sliced_data = list(islice(data_iter, start, stop if stop > 0 else None))
    except Exception as e:
        print(e)
        exit(1)

    for scene, files in sliced_data:
        print(f"Scene: {scene}")

        max_image_shape = None # Maximal size of the output image

        # Load file paths for the current scene
        if files[DATA_INFO_KEY]:
            with open(files[DATA_INFO_KEY], 'r') as f:
                texture_name = f.readline().strip()
                # Get the source and destination paths
                texture_path_src = os.path.join(TEXTURE_PATH, texture_name)
                texture_path_dst = os.path.join(OUTPUT_PATH, scene, texture_name)
                # Load the texture image
                texture_img = cv2.imread(texture_path_src)
                print("   Texture:", texture_name, "Shape:", texture_img.shape)
                # Copy the texture to the target location
                os.makedirs(os.path.join(OUTPUT_PATH, scene), exist_ok=True)
                shutil.copy(texture_path_src, texture_path_dst)

                max_image_shape = texture_img.shape
        else:
            # Data info not found
            continue

        # UV mapping
        if(TOGGLE_UV_OR_HGOF == "UV"):
            for i, view_img_path in enumerate(files[VIEW_IMGS_DIR]):
                # The UV map for the specific view_img_path
                uv_img_path = files[UV_IMGS_DIR][i]

                print("   Image: ", view_img_path)
                print("   UV img:", uv_img_path)

                aligned_img = align_image_uv(max_image_shape, view_img_path,
                    uv_img_path, uv_upscale_factor=UV_UPSCALE_FACTOR)

                output_img_path = os.path.join(OUTPUT_PATH, scene,
                                               VIEW_IMGS_DIR,
                                               os.path.basename(view_img_path))
                save_img(aligned_img, output_img_path)
                print("   Saved image:", output_img_path, aligned_img.shape)

        # Homography + Optical flow
        else:
            print_ram()

            ref_view_img = None
            if(len(files[VIEW_IMGS_DIR]) > 1):
                # Reference top view is the first image in the directory
                ref_view_img_path = files[VIEW_IMGS_DIR][0]
                # Reference top view UV map used for crop keypoints
                ref_uv_img_path = files[UV_IMGS_DIR][0]

                ref_view_img = get_ref_image(max_image_shape,
                                             ref_view_img_path, ref_uv_img_path,
                                             mask_object=MASK_OBJECT)
                print("   Reference view:", os.path.basename(ref_view_img_path),
                      "Resized shape:", ref_view_img.shape)

                output_img_path = os.path.join(OUTPUT_PATH, scene,
                                               VIEW_IMGS_DIR,
                                               os.path.basename(ref_view_img_path))
                save_img(ref_view_img, output_img_path)
                print("   Saved image:", output_img_path, ref_view_img.shape)
            else:
                print("Not enough files")
                continue

            for i, view_img_path in enumerate(files[VIEW_IMGS_DIR][1:], start=1):
                # The UV map for the specific view_img_path
                uv_img_path = files[UV_IMGS_DIR][i]

                print("   Image: ", view_img_path)
                print("   UV img:", uv_img_path)

                aligned_img = align_image_hg_of(model, model_args, ref_view_img,
                                                view_img_path, uv_img_path,
                                                mask_object=MASK_OBJECT)

                output_img_path = os.path.join(OUTPUT_PATH, scene,
                                               VIEW_IMGS_DIR,
                                               os.path.basename(view_img_path))
                save_img(aligned_img, output_img_path)
                print("   Saved image:", output_img_path, aligned_img.shape)

                # Free up the memory
                print_ram()
                torch.cuda.empty_cache()
                gc.collect()
                print_ram()
