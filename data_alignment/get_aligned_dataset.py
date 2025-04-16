# get_aligned_dataset.py

import os
import gc
import cv2
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from image_utils import load_data_iteratively, save_img, plot_image, \
    load_copy_texture
from align_by_uv_mapping import align_image_uv
from align_by_homography_opticalflow import init_opticalflow_model, \
    get_ref_image, print_ram, align_image_hg_of_with_tiling

VIEW_IMGS_DIR = "color_imgs" # Name of a directory for view images
UV_IMGS_DIR = "uv_imgs"      # Name of a directory for UV maps
DATA_INFO_FILE = "data.txt"  # Name of a file with a source texture name


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
    parser.add_argument("--mask", action='store_true',
        help="Mask out alpha channels using UV data (remove a background)")
    parser.add_argument("--uv_upscale", type=float, default=1.0,
        help="UV map upscale factor compared to the texture size for UV mapping")
    parser.add_argument("--compression", type=int, default=6,
        help="PNG output image compression level 0-9")
    # Homography and optical flow
    parser.add_argument("--model_name", type=str, default="sea_raft_s",
        help="Name of the optical flow model to use (default: sea_raft_s)")
    parser.add_argument("--ckpt_path", type=str, default="kitti",
        help="Path to the model checkpoint (default: kitti)")
    parser.add_argument("--patch_size", type=int, default=3072,
        help="Size of a patch for optical flow")
    parser.add_argument("--patch_stride", type=int, default=2048,
        help="Size of a patch stride for optical flow")
    parser.add_argument("--max_size_hg", type=int, default=1024,
        help="Maximal side size of the input image for homography (in pixels)")
    parser.add_argument("--max_size_of", type=int, default=1024,
        help="Maximal side size of the input image to the SEA-RAFT (in pixels)")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    DATA_PATH = args.data_path       # Path to a data directory 
    TEXTURE_PATH = args.texture_path # Path to a texture directory
    OUTPUT_PATH = args.output_path   # Path to an output directory
    TOGGLE_UV_OR_HGOF = 'HGOF' if args.hgof else 'UV'
    # For homography and optical flow alignment the UV maps are not needed
    if(TOGGLE_UV_OR_HGOF == 'HGOF'):
        UV_IMGS_DIR = None

    if None in [DATA_PATH, TEXTURE_PATH, OUTPUT_PATH]:
        raise ValueError("One or more required paths are not set")

    # Initialization of an optical flow SEA-RAFT model
    model = init_opticalflow_model(
        model_name=args.model_name, ckpt_path=args.ckpt_path
    )

    for scene, files in load_data_iteratively(DATA_PATH, VIEW_IMGS_DIR,
        UV_IMGS_DIR, DATA_INFO_FILE, DATA_INFO_FILE):
        print(f"Scene: {scene}")

        max_image_shape = None # Maximal size of the output image

        # Load a texture image shape for the current scene
        if files[DATA_INFO_FILE]:
            max_image_shape, texture_img_path = load_copy_texture(
                files[DATA_INFO_FILE], scene, TEXTURE_PATH, OUTPUT_PATH
            )
        else:
            # Data info not found
            print(f"{files[DATA_INFO_FILE]} not found")
            continue

        # UV mapping
        if(TOGGLE_UV_OR_HGOF == "UV"):
            for i, view_img_path in enumerate(files[VIEW_IMGS_DIR]):
                # The UV map for the specific view_img_path
                uv_img_path = files[UV_IMGS_DIR][i]

                print("Image: ", view_img_path)
                print("UV img:", uv_img_path)

                aligned_img = align_image_uv(max_image_shape, view_img_path,
                    uv_img_path, uv_upscale_factor=args.uv_upscale)

                output_img_path = os.path.join(
                    OUTPUT_PATH, scene, VIEW_IMGS_DIR,
                    os.path.basename(view_img_path))
                save_img(aligned_img, output_img_path,
                    compression_level=args.compression)
                print("Saved image:", output_img_path, aligned_img.shape)

        # Homography + Optical flow
        else:
            ref_view_img = None
            if(len(files[VIEW_IMGS_DIR]) > 1):
                # Reference top view
                # The texture image (all views aligned to a GT texture)
                ref_view_img_path = texture_img_path
                # Or the first image in the directory
                # ref_view_img_path = files[VIEW_IMGS_DIR][0]

                ref_view_img = get_ref_image(
                    max(max_image_shape[:2]), ref_view_img_path,
                    mask_object=args.mask)
                print("Reference view:", os.path.basename(ref_view_img_path),
                      "Resized shape:", ref_view_img.shape)

                output_img_path = os.path.join(
                    OUTPUT_PATH, scene, VIEW_IMGS_DIR,
                    os.path.basename(ref_view_img_path))
                # No need to save the reference texture as a view
                # save_img(ref_view_img, output_img_path,
                #     compression_level=args.compression)
                # print("Saved image:", output_img_path, ref_view_img.shape)
            else:
                print("Not enough files")
                continue

            for i, view_img_path in enumerate(files[VIEW_IMGS_DIR]):
                print("Image: ", view_img_path)

                aligned_img = align_image_hg_of_with_tiling(
                    model, args, ref_view_img, view_img_path,
                    max_size_hg=args.max_size_hg, max_size_of=args.max_size_of,
                    patch_size=args.patch_size, patch_stride=args.patch_stride)

                output_img_path = os.path.join(
                    OUTPUT_PATH, scene, VIEW_IMGS_DIR,
                    os.path.basename(view_img_path))
                save_img(aligned_img, output_img_path,
                    compression_level=args.compression)
                print("Saved image:", output_img_path, aligned_img.shape)

                # Free up the memory
                torch.cuda.empty_cache()
                gc.collect()
                # print_ram()
