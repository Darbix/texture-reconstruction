# align_photos.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cv2
import argparse

from image_utils import save_img
from align_by_homography_opticalflow import init_opticalflow_model, \
    align_image_hg_of_with_tiling, get_ref_image


def parse_arguments():
    """Argument parser"""
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("--scene_path", type=str,
        help="Path to the input views")
    parser.add_argument("--output_path", type=str,
        help="Path to the output data directory")
    parser.add_argument("--compression", type=int, default=6,
        help="PNG output image compression level 0-9")
    # Homography and optical flow
    parser.add_argument("--model_name", type=str, default="sea_raft_s",
        help="Name of the optical flow model to use (default: sea_raft_s)")
    parser.add_argument("--ckpt_path", type=str, default="kitti",
        help="Path to the model checkpoint (default: kitti)")
    parser.add_argument("--patch_size", type=int, default=1024,
        help="Size of a patch for optical flow")
    parser.add_argument("--patch_stride", type=int, default=512,
        help="Size of a patch stride for optical flow")
    parser.add_argument("--max_size_hg", type=int, default=2048,
        help="Maximal side size of the input image for homography (in pixels)")
    parser.add_argument("--max_size_of", type=int, default=1024,
        help="Maximal side size of the input image to the SEA-RAFT (in pixels)")
    parser.add_argument("--max_image_size", type=int, default=-1,
        help="Maximal side size of output images")
    return parser.parse_args()


def main():
    """
    Aligns images in a given directory to the first view in a specific
    resolution using homography and a specified optical flow model. The optical
    flow is done using tiling to keep the original image quality.
    """
    # Parse arguments
    args = parse_arguments()

    # Initialization of an optical flow SEA-RAFT model
    model = init_opticalflow_model(
        model_name=args.model_name, ckpt_path=args.ckpt_path
    )

    # All image file names
    image_files = sorted([f for f in os.listdir(args.scene_path) if f.endswith(('.jpg', '.png'))])
    
    # The first file is a reference image
    ref_img_file = image_files[0]
    ref_view_img_path = os.path.join(args.scene_path, ref_img_file)
    max_image_size = args.max_image_size
    if(max_image_size <= 0):
        ref_view_img = cv2.imread(ref_view_img_path)
        max_image_size = max(*ref_view_img.shape[:2])

    # Load the reference image and resize to an output size
    ref_view_img = get_ref_image(max_image_size, ref_view_img_path)

    # Save the reference view image
    ref_output_path = os.path.join(
        args.output_path, f"{os.path.splitext(ref_img_file)[0]}.jpg"
    )
    save_img(ref_view_img, ref_output_path, compression_level=args.compression)
    print("Reference view saved: ", ref_output_path)

    # Other views are aligned to the reference one
    for idx, img_file in enumerate(image_files[1:], start=1):
        view_img_path = os.path.join(args.scene_path, img_file)
        print(f"Loading a view {view_img_path}")
        
        # Align view by homography and optical flow
        aligned_img = align_image_hg_of_with_tiling(
            model, args, ref_view_img, view_img_path,
            max_size_hg=args.max_size_hg, max_size_of=args.max_size_of,
            patch_size=args.patch_size, patch_stride=args.patch_stride)

        # Save the aligned view image
        output_img_path = os.path.join(
            args.output_path, f"{os.path.splitext(img_file)[0]}.jpg"
        )
        save_img(aligned_img, output_img_path,
            compression_level=args.compression)
        print(f"Saved {output_img_path}")


if __name__ == "__main__":
    main()
