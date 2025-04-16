# extract_patches.py

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import cv2
import math
import json
import time
import random
import argparse
import numpy as np

from degradation import apply_strong_degradation, apply_light_degradation, \
    set_cfg

from utils import get_crop_positions, crop_patch, get_transparency_perc, \
    create_info_json, crop_patches

VIEWS_DIR = 'color_imgs'                    # Name of a subdir with input views
IMAGE_PATCHES_DIR = 'image_patches'         # Name of an output dir for image patches
TEXTURE_PATCHES_DIR = 'texture_patches'     # Name of an output dir for texture patches
REFERENCE_PATCHES_DIR = 'reference_patches' # Name of an output dir for reference patches


def parse_arguments():
    """Command line argument parser"""
    parser = argparse.ArgumentParser(description="Crop image patches from dataset.")
    parser.add_argument("--data_dir", type=str,
        help="Path to input data directory", required=True)
    parser.add_argument("--output_dir", type=str,
        help="Path to output directory", required=True)
    parser.add_argument("--patch_size", type=int, default=512,
        help="Size of patches")
    parser.add_argument("--patch_stride", type=int, default=512,
        help="Stride for cropping patches")
    parser.add_argument("--alpha_threshold", type=float, default=0.90,
        help="Transparency threshold")
    parser.add_argument("--quality", type=int, default=100, help="JPEG quality")
    parser.add_argument("--degradation_config", type=str,
        help="Path to YAML degradation config file")
    return parser.parse_args()


def main(data_dir, output_dir, patch_size=512,
    patch_stride=512, alpha_threshold=1.00, quality=100):
    """
    Processes and degrades image and crops them to patches.
    Realistically degraded reference views are created from the textures,
    which are located directly in particular scene directories.
    """
    patch_count = 0  # Number of patches processed
    saved_count = 0  # Number of patches saved (transparent are skipped)

    texture_w = None  # GT texture width
    texture_h = None  # GT texture height
    texture_pw = None # GT texture padded width
    texture_ph = None # GT texture padded height
    view_w = None     # View image width 
    view_h = None     # View image height 
    view_pw = None    # View image padded width 
    view_ph = None    # View image padded height

    # Scene directory names
    scene_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(
        os.path.join(data_dir, d))])

    for dir_name in scene_dirs:
        scene_dir_path = os.path.join(data_dir, dir_name)
        start_time = time.time()
        
        # Traverse all subdirectories and files using os.walk
        for subdir, _, files in os.walk(scene_dir_path):
            files = sorted(files)

            for file_name in files:
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Original file path
                    old_path = os.path.join(subdir, file_name)

                    # Compute relative path to keep structure
                    rel_path = os.path.relpath(subdir, data_dir)

                    # Modify the dataset structure
                    is_texture = False
                    # Dir for view patches
                    if rel_path.endswith(VIEWS_DIR):
                        new_dir = os.path.join(output_dir,
                            rel_path.replace(VIEWS_DIR, IMAGE_PATCHES_DIR))
                    # Dir for texture patches
                    else:
                        is_texture = True
                        new_dir = os.path.join(output_dir, rel_path,
                            TEXTURE_PATCHES_DIR)

                    os.makedirs(new_dir, exist_ok=True)

                    # Load an image
                    img = cv2.imread(old_path, cv2.IMREAD_UNCHANGED)
                    if img.shape[-1] != 4:
                        # Add alpha channel (RGB -> RGBA)
                        alpha_channel = np.ones((img.shape[0], img.shape[1], 1),
                                dtype=img.dtype) * 255
                        img = np.concatenate([img, alpha_channel], axis=-1)

                    height, width = img.shape[:2]
                
                    # Get crop positions
                    crop_positions, ph, pw = get_crop_positions(
                        height, width, patch_size, patch_stride)
                    print(f"Processing {old_path} with {len(crop_positions)} patches")

                    def call_crop_patches():
                        nonlocal patch_count, saved_count
                        # Process the image to patches
                        temp_patch_count, temp_saved_count = crop_patches(
                            crop_positions, img, patch_size,
                            alpha_threshold, quality, file_name, new_dir)
                        patch_count += temp_patch_count
                        saved_count += temp_saved_count

                    if(is_texture):
                        # Process a texture image
                        call_crop_patches()

                        # If the image is a texture, a degraded version of it is
                        # used as a reference view
                        new_dir = os.path.join(output_dir, rel_path,
                            REFERENCE_PATCHES_DIR)
                        os.makedirs(new_dir, exist_ok=True)
                        # Process a reference view image (degraded texture)
                        img = apply_strong_degradation(img)
                        call_crop_patches()

                        texture_w = width
                        texture_h = height
                        texture_pw = pw
                        texture_ph = ph
                    else:
                        # Process a view image
                        img = apply_light_degradation(img)
                        call_crop_patches()

                        view_w = width
                        view_h = height
                        view_pw = pw
                        view_ph = ph

        # Create info.json for each processed directory
        create_info_json(os.path.join(output_dir, dir_name), patch_size,
                         patch_stride, texture_w, texture_h, texture_pw,
                         texture_ph, view_w, view_h, view_pw, view_ph)

        elapsed_time = time.time() - start_time
        print(f"Processed {dir_name} in {elapsed_time:.2f} seconds\n")

    print(f"Total patches processed: {patch_count}")
    print(f"Total patches saved: {saved_count}")



if __name__ == "__main__":
    args = parse_arguments()

    set_cfg(args.degradation_config)

    main(args.data_dir, args.output_dir, args.patch_size,
        args.patch_stride, args.alpha_threshold, args.quality)
