# image_utils.py

import os
import cv2
import shutil
import OpenEXR
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def load_data_iteratively(data_path, VIEW_IMGS_DIR, UV_IMGS_DIR, DATA_INFO_FILE,
    DATA_INFO_KEY):
    """Loads file paths for each data set scene by scene"""
    for scene in os.listdir(data_path):
        # Path to UID specific set of data views
        scene_path = os.path.join(data_path, scene)
        if not os.path.isdir(scene_path):
            continue

        # Get all paths to images and other data for a specific scene
        files = {
            VIEW_IMGS_DIR: sorted(glob(os.path.join(scene_path, VIEW_IMGS_DIR, "*.png")) + glob(os.path.join(scene_path, VIEW_IMGS_DIR, "*.jpg"))),
            UV_IMGS_DIR:   sorted(glob(os.path.join(scene_path, UV_IMGS_DIR, "*.exr"))) if UV_IMGS_DIR else None,
            DATA_INFO_KEY: os.path.join(scene_path, DATA_INFO_FILE) if os.path.exists(os.path.join(scene_path, DATA_INFO_FILE)) else None
        }

        yield scene, files


def load_copy_texture(info_file_path, scene_dir_name, texture_path, output_path):
    """Loads a scene info file to get a texture name and copies the file"""
    with open(info_file_path, 'r') as f:
        texture_name = f.readline().strip()
        # Get the source and destination paths
        texture_path_src = os.path.join(texture_path, texture_name)
        texture_path_dst = os.path.join(output_path, scene_dir_name, texture_name)
        # Load the texture image
        texture_img = cv2.imread(texture_path_src)
        print("Texture:", texture_name, "Shape:", texture_img.shape)
        # Copy the texture to the target location
        os.makedirs(os.path.join(output_path, scene_dir_name), exist_ok=True)
        shutil.copy(texture_path_src, texture_path_dst)

        return texture_img.shape, texture_path_src


def save_img(aligned_img, img_path, compression_level=0):
    """Saves an image with PNG compression and creates directories on the path"""
    dir_name = os.path.dirname(img_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    
    # Set compression level (0 = no compression, 9 = maximum compression)
    compression_params = [cv2.IMWRITE_PNG_COMPRESSION, compression_level]
    
    cv2.imwrite(img_path, aligned_img, compression_params)


def find_bounding_box(alpha_channel):
    """Finds the bounding box of nonzero alpha values"""
    non_zero_alpha = alpha_channel > 0.0
    if np.any(non_zero_alpha):
        min_y, min_x = np.min(np.nonzero(non_zero_alpha), axis=1)
        max_y, max_x = np.max(np.nonzero(non_zero_alpha), axis=1)
        return (min_x, min_y, max_x, max_y)
    else:
        return None


def crop_image_to_bbox(img, bounding_box):
    """Crops an image to the given bounding box"""
    if bounding_box:
        min_x, min_y, max_x, max_y = bounding_box
        cropped_img = img[min_y:max_y, min_x:max_x]
        return cropped_img
    else:
        return img


def load_exr_to_array(exr_filename):
    """Loads an EXR image into a NumPy array."""
    exr_file = OpenEXR.InputFile(exr_filename)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = exr_file.channels(["R", "G", "B", "A"])
    r, g, b, a = (np.frombuffer(ch, dtype=np.float16).reshape((height, width)) for ch in channels)
    return np.stack((r, g, b, a), axis=-1)


def resize_to_max_size(image, max_size=-1, interpolation=cv2.INTER_AREA, 
    clip_range=None):
    """
    Resizes a NumPy image to a specific max size of the longer side
    and keeps aspect ratio.
    """
    if max_size <= 0:
        return image
    h, w = image.shape[:2]
    # Scale factor for the longer side
    scale = max_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    if clip_range:
        image = np.clip(image, *clip_range)
    return image


def plot_image(image):
    """Plot a numpy image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.max() > 1:
        image = image / 255.0

    plt.imshow(image)
    plt.show()
