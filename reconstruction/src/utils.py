# utlis.py

import os
import re
import cv2
import math
import json
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

import config


def get_patch_weight_blend_mask(patch_size):
    """Creates a mask of weights for gradient blend of patches"""
    # Auxiliary variables to create a gradient mask
    y_grid = np.linspace(0, 2, patch_size).reshape(-1, 1)
    x_grid = np.linspace(0, 2, patch_size).reshape(1, -1)
    # Mask in range of values (0.0 - 1.0) with higher weight in the center
    blend_mask = np.minimum(np.minimum(y_grid, 2 - y_grid),
                            np.minimum(x_grid, 2 - x_grid))
    blend_mask = np.expand_dims(blend_mask, axis=-1) # [H, W, 1]
    
    # To strenghten the patch center weight over the edges
    # blend_mask = np.sqrt(blend_mask)

    blend_mask = np.clip(blend_mask, 1e-7, None)

    return blend_mask


def attach_patch_to_image(
        composed_image, weight_map, img_size=None, sr_factor=1,
        patch=None, patch_x=None, patch_y=None, blend_mask=None):
    """Attaches a patch ([C, H, W] tensor) to the NumPy image at [x, y]"""
    # Square patch size
    patch_size = patch.shape[1]
    # Convert pytorch tensor [C, H, W] to numpy [H, W, C]
    patch = patch.cpu().detach().numpy().transpose(1, 2, 0)  

    # Resize the patch and adapt a position using sr_factor
    if sr_factor > 1:
        patch_x = int(patch_x * sr_factor)
        patch_y = int(patch_y * sr_factor)
        img_size = (img_size[0], img_size[1] * sr_factor, img_size[2] * sr_factor)

    # Initialize for the first patch
    if composed_image is None:
        composed_image = np.zeros((img_size[1], img_size[2], img_size[0]), dtype=np.float32)
    if weight_map is None:
        weight_map = np.zeros((img_size[1], img_size[2], 1), dtype=np.float32)

    if(blend_mask is None):
        # Default mask where all pixels in all patches are equal
        blend_mask = np.ones((patch_size, patch_size, 1))
    
    # # Normalize [-1.0, 1.0] to [0.0, 1.0]
    patch = normalize_image(patch)

    # Add weighted patch to the composed image
    composed_image[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size, :] += patch * blend_mask
    # Accumulated weights for a final normalization of composed patches
    weight_map[patch_y:patch_y + patch_size, patch_x:patch_x + patch_size, :] += blend_mask
    
    return composed_image, weight_map


def normalize_composed_image(composed_image, weight_map):
    """Normalizes the image to [0.0, 1.0] using the accumulated weight map"""
    # Avoid a division by zero
    weight_map = np.clip(weight_map, 1e-7, None)
    return (composed_image / weight_map)


def normalize_image(image):
    """Normalize an array from [-1.0, 1.0] to [0.0, 1.0]"""
    return np.clip((image + 1) / 2, 0.0, 1.0)


def plot_patches(path, b_lr_patches, b_hr_texture_patch, b_sr_img_patch):
    """Plot and save LR view patches, GT texture patch and an output patch"""
    n_patches = b_lr_patches.shape[0] # Number of view patches
    
    # Determine number of columns based on n_patches (fixed 2 rows)
    n_cols = (n_patches + 2 + 1) // 2
    fig, axes = plt.subplots(2, n_cols, figsize=(15, 5))

    # Flatten axes array for easier indexing
    axes = axes.flatten()

    # Plot the view patches
    for i in range(min(n_patches, len(axes) - 2)):
        img = b_lr_patches.cpu().numpy().transpose(0, 2, 3, 1)[i]
        img = normalize_image(img)
        axes[i].imshow(img)
        axes[i].set_title(f"Image {i+1}")

    # Plot the texture patch
    img = b_hr_texture_patch.cpu().numpy().transpose(1, 2, 0)
    img = normalize_image(img)
    axes[-2].imshow(img)
    axes[-2].set_title("Texture")

    # Plot the output patch
    img = b_sr_img_patch.cpu().detach().numpy().transpose(1, 2, 0)
    img = normalize_image(img)
    axes[-1].imshow(img)
    axes[-1].set_title("Output")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def get_patch_positions(ref_path):
    """
    Load a list of all patch positions using reference patches (files) in ref_path
    """
    filenames = os.listdir(ref_path)
    return [
        (string_to_position(filename)) for filename in filenames 
        if string_to_position(filename)
    ]


def load_img_patches(scene_path, n_patches, patch_positions):
    """Yields patches (sets of n_patches of views and 1 texture) on positions"""
    texture_patches_dir_path = os.path.join(scene_path, config.TEXTURE_PATCHES_DIR)
    view_patches_dir_path = os.path.join(scene_path, config.VIEW_PATCHES_DIR)
    ref_patches_dir_path = os.path.join(scene_path, config.REF_PATCHES_DIR)

    transform_patch = get_patch_transform()
    
    # Load scene info
    with open(os.path.join(scene_path, config.SCENE_INFO_DATA), 'r') as f:
        scene_info = json.load(f)

    # Save the total number of patches
    scene_info["len"] = len(patch_positions)

    for patch_y, patch_x in patch_positions:
        # Load a pair of a texture patch and N view patches
        texture_patch, view_patches = load_patch_set(
            (patch_y, patch_x), texture_patches_dir_path,
            ref_patches_dir_path, view_patches_dir_path, transform_patch,
            scene_info["patch_size"], n_patches
        )

        yield torch.stack(view_patches), texture_patch, (patch_y, patch_x), scene_info


def get_patch_transform():
    """Returns a transform for converting patches to tensors"""
    transform_patch = T.Compose([
        # T.Resize((512, 512)), # TODO debug
        T.ToTensor(),
        # Normalize to [-1, 1]
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return transform_patch


def process_image(path, max_res=None):
    """Loads an image (patch) from a path and eventually resizes it"""
    with Image.open(path) as img:
        if max_res > 0:
            return img.resize((max_res, max_res))
        else:
            img.load()
            return img


def string_to_position(filename):
    """Converts a string filename patch position to numbers"""
    match = re.search(r'y_(\d+)_x_(\d+)', filename)
    return tuple(map(int, match.groups())) if match else None


def position_to_string(y, x):
    """Converts patch position to a string"""
    return f"_y_{y}_x_{x}"


def load_patch_set(pos, texture_dir, ref_dir, view_dir, transform_patch,
    patch_size, n_patches, max_patch_size=-1):
    """
    Loads a pair of a texture patch and N view patches from files as tensors.
    The first LR view will be the reference one.
    """
    str_patch_pos = position_to_string(*pos)
    texture_patch = None
    view_patches_list = []

    # Get a texture patch
    for file_name in os.listdir(texture_dir):
        name, ext = os.path.splitext(file_name)
        if name.endswith(f"{str_patch_pos}"):
            patch = transform_patch(process_image(os.path.join(texture_dir, file_name), max_patch_size))
            texture_patch = patch

    # Get a reference patch
    for file_name in os.listdir(ref_dir):
        name, ext = os.path.splitext(file_name)
        if name.endswith(f"{str_patch_pos}"):
            patch = transform_patch(process_image(os.path.join(ref_dir, file_name), max_patch_size))
            view_patches_list.append(patch)

    # Shuffle views (TODO or some way to pick the best ones)
    view_patches_paths = os.listdir(view_dir)
    random.shuffle(view_patches_paths)
    
    # Load up to N view patches
    for file_name in view_patches_paths:
        name, ext = os.path.splitext(file_name)
        if name.endswith(f"{str_patch_pos}"):
            patch = transform_patch(process_image(os.path.join(view_dir, file_name), max_patch_size))
            view_patches_list.append(patch)
            if len(view_patches_list) >= n_patches:
                break
    
    # Fill missing patches with blank (black) images
    missing_patches = n_patches - len(view_patches_list)
    if missing_patches > 0:
        view_patches_list.extend(
            [transform_patch(
                Image.new('RGB', (patch_size, patch_size), color=(0, 0, 0))
            )] * missing_patches
        )
    
    return texture_patch, view_patches_list


@torch.no_grad
def compose_sr_lr_hr(model, data_path, n_patches, max_size=-1):
    """
    Enhances and composes images from patches in a specified directory
    structure data_path/<texture_patches, reference_patches, image_patches> with
    data_path/info.json scene and patch information.
    """
    blend_mask = None # Mask for blending patches (fades overlapped patches)
    # The blending masks accumulate to weight maps
    view_image, view_weight_map = None, None       # Reference view image
    texture_image, texture_weight_map = None, None # GT texture image 
    output_image, output_weight_map = None, None   # Result output image
    # Path to patches from which the positions extract
    ref_path = os.path.join(data_path, config.TEXTURE_PATCHES_DIR)

    i = 0
    # Generate sets of patches
    for view_patches, texture_patch, patch_pos, scene_info in load_img_patches(
        data_path, n_patches, get_patch_positions(ref_path)):

        b_output_patch = model(view_patches.unsqueeze(0))

        patch_y, patch_x = patch_pos
        img_size = (
            view_patches.shape[1], 
            scene_info["texture_ph"],
            scene_info["texture_pw"]
        )
        
        # Create the blend mask only once (LR, texture and output patches must have the same size)
        if(blend_mask is None):
            patch_size = b_output_patch[0].shape[1]
            blend_mask = get_patch_weight_blend_mask(patch_size)
        
        # Output image
        output_image, output_weight_map = attach_patch_to_image(
            output_image, output_weight_map, img_size=img_size, patch=b_output_patch[0],
            patch_x=patch_x, patch_y=patch_y, blend_mask=blend_mask
        )
        # First view (reference) input image
        view_image, view_weight_map = attach_patch_to_image(
            view_image, view_weight_map, img_size=img_size, patch=view_patches[0],
            patch_x=patch_x, patch_y=patch_y, blend_mask=blend_mask
        )
        # Texture GT image
        texture_image, texture_weight_map = attach_patch_to_image(
            texture_image, texture_weight_map, img_size=img_size, patch=texture_patch,
            patch_x=patch_x, patch_y=patch_y, blend_mask=blend_mask
        )

        i += 1
        if i > 5:
            break
        total_patches = scene_info["len"]
        print(f"{i}/{total_patches} patches processed")

    # Normalize images by weight maps
    output_image = normalize_composed_image(output_image, output_weight_map)
    view_image = normalize_composed_image(view_image, view_weight_map)
    texture_image = normalize_composed_image(texture_image, texture_weight_map)

    # Resize images to not be larger than max_size
    output_image = resize_to_max_size((output_image), max_size, clip_range=(0.0, 1.0))
    view_image = resize_to_max_size((view_image), max_size, clip_range=(0.0, 1.0))
    texture_image = resize_to_max_size((texture_image), max_size, clip_range=(0.0, 1.0))

    return output_image, view_image, texture_image


def resize_to_max_size(image, max_size=-1, clip_range=None):
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
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if clip_range:
        image = np.clip(image, *clip_range)
    return image


def plot_random_val_patch(model, val_dataloader, val_path, device):        
    """Randomly select a patch, apply the model and save the result image"""
    # Load a random sample from a dataloader
    random_idx = random.randint(0, len(val_dataloader.dataset) - 1)
    val_lr_patches, val_hr_texture_patch, _ = val_dataloader.dataset[random_idx]
    # Add batch dimension and convert to device
    val_lr_patches = val_lr_patches.unsqueeze(0).to(device, non_blocking=True)
    val_hr_texture_patch = val_hr_texture_patch.unsqueeze(0).to(device, non_blocking=True)

    val_sr_patch = model(val_lr_patches)

    # Take a first patch set from the batch
    batch_i = 0
    # Plot a reference LR, an HR texture and a SR output
    plot_patches(val_path, 
                val_lr_patches[batch_i], 
                val_hr_texture_patch[batch_i], 
                val_sr_patch[batch_i])
