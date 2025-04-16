# utils.py

import os
import cv2
import math
import json
import numpy as np


def get_crop_positions(height, width, patch_size, patch_stride):
    """Generates all possible crop positions for patches within an image"""
    crop_positions = []
    padded_height = math.ceil(height / patch_size) * patch_size
    padded_width = math.ceil(width / patch_size) * patch_size

    # Generate all patch positions for cropping
    for i in range(0, padded_height + 1 - patch_size, patch_stride):
        for j in range(0, padded_width + 1 - patch_size, patch_stride):
            crop_positions.append((i, j))

    return crop_positions, padded_height, padded_width


def crop_patch(img, patch_x, patch_y, patch_size):
    """Extracts a patch from a full-size image, padding if needed"""
    patch = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)
    cropped = img[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size]
    
    # Place the cropped image into the patch
    patch[0:cropped.shape[0], 0:cropped.shape[1]] = cropped

    return patch


def get_transparency_perc(patch):
    """Checks if more than a certain percentage of the pixels are fully transparent"""
    alpha_channel = patch[:, :, 3]  # Extract alpha channel
    transparent_pixels = np.sum(alpha_channel < 128)
    transparency_percentage = transparent_pixels / alpha_channel.size
    return transparency_percentage


def create_info_json(dir_name, patch_size, stride_size, texture_w, texture_h,
    texture_pw, texture_ph, view_w, view_h, view_pw, view_ph):
    """Create a json file with scene and and patch general information"""
    # Define the information to be written to the JSON file
    info = {
        "patch_size": patch_size,
        "stride_size": stride_size,
        "texture_w": texture_w,
        "texture_h": texture_h,
        "texture_pw": texture_pw,
        "texture_ph": texture_ph,
        "view_w": view_w,
        "view_h": view_h,
        "view_pw": view_pw,
        "view_ph": view_ph
    }

    # Create the directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Path for the info.json file
    info_json_path = os.path.join(dir_name, "info.json")

    # Write the information to the JSON file
    with open(info_json_path, 'w') as json_file:
        json.dump(info, json_file, indent=4)

    print(f"info.json created at: {info_json_path}")


def crop_patches(crop_positions, img, patch_size, alpha_threshold, quality,
    file_name, new_dir):
    """Given specific patch position pairs, crops an image into patches"""
    patch_count = 0
    saved_count = 0

    for y, x in crop_positions:
        patch = crop_patch(img, x, y, patch_size)

        # Check if the patch is transparent enough (or almost black with mean
        # pixel value under 5) to be skipped
        transparency_perc = get_transparency_perc(patch)
        black_mean = patch[:, :, :3].mean()
        if(transparency_perc >= alpha_threshold or black_mean < 5):
            print(f"Skipping patch at x: {x:4}, y: {y:4} due to transparency",
                  f"{transparency_perc:.2f}. Black level: {black_mean:.2f}")
            patch_count += 1
            continue

        # New file name
        name, _ = os.path.splitext(file_name)
        new_name = f"{name}_y_{y}_x_{x}.jpg"
        new_path = os.path.join(new_dir, new_name)

        # Save as high-quality JPG
        cv2.imwrite(new_path, patch[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, quality])

        print(f"Saved patch: {new_path}")

        patch_count += 1
        saved_count += 1
    return patch_count, saved_count
