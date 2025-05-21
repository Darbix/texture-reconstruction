# test.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config')))

import cv2
import json
import math
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import torch
import torch.utils.data as data
import torchvision.transforms as T

from model_utils import load_checkpoint, setup_model
from utils import resize_to_max_size, plot_patches, get_patch_transform, \
    get_patch_weight_blend_mask, attach_patch_to_image, normalize_composed_image
import config


TEST_IMGS_SUBDIR = 'color_imgs' # Subdir with images in each '--data_path' scene dir for group evaluation

def parse_args():
    parser = argparse.ArgumentParser(description="Enhance LR image using multi-views and a trained model")
    # Paths
    parser.add_argument('--data_path', type=str, required=False, help="Path to the dataset for evaluation")
    parser.add_argument('--imgs_path', type=str, required=False, help="Path to the image views for enhancement")
    parser.add_argument('--checkpoint_path', type=str, required=True, help="Path to model checkpoint file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to the output image")
    # Model attributes
    parser.add_argument('--model_type', type=str, default='UNET', help="'UNET' or 'DEFAULT' MVTRN")
    parser.add_argument('--num_views', type=int, default=6, help="Number of views in a patch set")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers loading images")
    # Image and patches
    parser.add_argument('--patch_size', type=int, default=512, help="Patch size")
    parser.add_argument('--patch_stride', type=int, default=-1, help="Patch stride")
    parser.add_argument('--alpha_threshold', type=float, default=0.9, help="Maximum transparency to keep a patch")
    parser.add_argument('--max_image_size', type=int, default=-1, help="Maximum size for the output image")
    # Other
    parser.add_argument('--device', type=str, default='cuda', help="Device to run the model on (cpu or cuda)")
    parser.add_argument('--gt_texture', type=str, help="Path to a GT texture to compare with a reconstructed image")
    return parser.parse_args()


def compare_images(gt_texture, image, metric='PSNR'):
    """Compares uint8 images by a specific metric"""
    if(gt_texture.shape[:2] != image.shape[:2]):
        print(f"The image shapes do not match {gt_texture.shape[:2]} != {image.shape[:2]}",
            "These images cannot be compared.")
        return None
    
    if(metric == 'PSNR'):
        psnr_value = psnr(gt_texture, image)
        return psnr_value
    elif(metric == 'SSIM'):
        ssim_value = ssim(gt_texture, image, channel_axis=-1)
        return ssim_value

    return None


def save_lr_ref_view_resized(image_shape, ref_image_path, output_path):
    """
    Saves the input LR reference view as the visual reference image to the
    enhanced image by multiple views. The image is resized to the same shape.
    """
    image_h, image_w = image_shape

    # Load an RGB ref view image and resize to fit the specific max image size
    ref_image = load_image(ref_image_path, max(image_shape))[:, :, :3]

    plt.imsave(output_path, ref_image)
    print(f"The reference image saved to {output_path}")


def load_image(image_path, max_image_size):
    """Loads and processes (resizes) a single NumPy image from the path"""
    with Image.open(image_path) as img:
        img = img.convert('RGB') # Removes possible alpha channel
        img = np.array(img)
        img = resize_to_max_size(img, max_image_size)
        return img


def load_images(dir_path, image_files, max_image_size=-1, num_workers=1):
    """
    Loads all images to a list from a specific dir_path using a given
    number of workers. Images may eventually be resized if max_image_size > 0.
    """
    images = []
    if num_workers == 1:
        # Sequential loading
        for image_file in image_files:
            image_path = os.path.join(dir_path, image_file)
            images.append(load_image(image_path, max_image_size))
    else:
        # Parallel loading
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for image_file in image_files:
                image_path = os.path.join(dir_path, image_file)
                futures.append(executor.submit(load_image, image_path, max_image_size))

            for future in futures:
                images.append(future.result())
    return images


def get_transparency_perc(patch):
    """Counts the percentage of pixels that are fully transparent"""
    if patch.shape[-1] < 4:
        raise ValueError("Input image does not have an alpha channel.")
    # Extract the alpha channel (last channel in RGBA format)
    alpha_channel = patch[:, :, 3]
    # Count fully transparent pixels (alpha == 0)
    transparent_pixels = np.sum(alpha_channel == 0)
    # Calculate transparency percentage
    total_pixels = alpha_channel.size
    transparency_percentage = transparent_pixels / total_pixels
    return transparency_percentage


@torch.no_grad
def enhance_image_multiview(model, data_path, output_path, num_views,
    patch_size, patch_stride, max_image_size, transform_patch=None,
    alpha_threshold=0.9, num_workers=1, device='cpu'):
    """
    Enhances a LR reference image using multi-view images from a given data
    directory. In the alphabetical sort of files, the first image is considered
    the reference view.
    """
    # Init variables
    transform_patch = get_patch_transform() if not transform_patch else transform_patch
    patch_stride = patch_size if patch_stride <= 0 else patch_stride

    # Load and sort all images so the first image is the reference one
    image_files = [f for f in os.listdir(data_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files)
    # List of all views
    images = load_images(
        data_path, image_files, max_image_size=max_image_size,
        num_workers=num_workers
    )
    # Assuming all images are the same size!
    orig_img_height, orig_img_width = images[0].shape[:2]
    # Image size with padding caused by patches and stride
    img_height = math.ceil((orig_img_height - patch_size) / patch_stride) * patch_stride + patch_size
    img_width = math.ceil((orig_img_width - patch_size) / patch_stride) * patch_stride + patch_size

    # Total number of patches
    total_patches_num = ((img_height - patch_size) // patch_stride + 1) * ((img_width - patch_size) // patch_stride + 1)
    
    n_patches_processed = 0 # Number of patches processed
    view_image = None       # Composed LR image view
    output_image = None     # Composed output image
    weight_map = None       # Weight mask to weight patches with masks for blending
    blend_mask = get_patch_weight_blend_mask(patch_size) # Mask for blending patches

    # Loop over all positions (y, x) and load patch sets
    for y in range(0, img_height - patch_size + 1, patch_stride):
        for x in range(0, img_width - patch_size + 1, patch_stride):
            patch_set = [] # Current position set of patches
            
            # Loop over all loaded images and crop patches at (x, y)
            for i, img_array in enumerate(images):
                # No more patches needed
                if(len(patch_set) >= num_views):
                    break

                # Crop the patch from the numpy image
                patch = img_array[y:y + patch_size, x:x + patch_size]
                
                # If the patch is RGBA, check the amount of transparency
                if(patch.shape[2] > 3):
                    transparency_perc = get_transparency_perc(patch)

                    # Skip the patch if it's almost empty and enough patches remain
                    if(transparency_perc >= alpha_threshold 
                        and len(images[i+1:]) > (num_views - len(patch_set))):
                        continue

                # Convert to RGB for the rest of processing
                patch = patch[:, :, :3]

                # Pad a non-square patch
                if patch.shape[:2] != (patch_size, patch_size):
                    padding = ((0, patch_size - patch.shape[0]),
                               (0, patch_size - patch.shape[1]), (0, 0))
                    patch = np.pad(patch, padding, mode='constant', constant_values=0)
                
                patch_set.append(transform_patch(patch))
            
            # Add blank patches to get exactly 'num_views' patches in the set
            missing_patch_num = num_views - len(patch_set)
            if(missing_patch_num > 0):
                patch_set.extend(
                    [transform_patch(
                        Image.new('RGB', (patch_size, patch_size), color=(0, 0, 0))
                    )] * missing_patch_num
                )

            # Enhance the patch using the model
            b_output_patch = model(torch.stack(patch_set).unsqueeze(0).to(device))

            # Attach the enhanced patch to the output image
            output_image, weight_map = attach_patch_to_image(
                output_image, weight_map, img_size=(3, img_height, img_width),
                patch=b_output_patch[0], patch_x=x, patch_y=y,
                blend_mask=blend_mask
            )

            n_patches_processed += 1

        print(f"{n_patches_processed}/{total_patches_num} patches processed")

    # Normalize the composed image and blend patches correctly
    image = normalize_composed_image(output_image, weight_map)

    # Cut off the padding
    image = image[:orig_img_height, :orig_img_width, :]

    if(output_path):
        output_dir = os.path.dirname(output_path)
        if output_dir: # Only create if there is a directory part
            os.makedirs(output_dir, exist_ok=True)
        plt.imsave(output_path, image)
        print(f"The output enhanced image saved to {output_path}")

    return image



if __name__ == "__main__":
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading a model")
    model = setup_model(args.model_type, num_views=args.num_views)
    model = model.to(device)

    print(f"Loading checkpoint (model weights) from {args.checkpoint_path}")
    optimizer = None
    model, optimizer, epoch, loss_hist = load_checkpoint(
        model, args.checkpoint_path, optimizer, device)
    model.eval()

    list_img_dirs = []    # List of paths to scene directories
    list_gt_textures = [] # List of paths to GT textures
    if args.imgs_path:
        list_img_dirs = [args.imgs_path]
        list_gt_textures = [args.gt_texture]
    else:
        for scene_dir in sorted(os.listdir(args.data_path)):
            scene_path = os.path.join(args.data_path, scene_dir)
            if os.path.isdir(scene_path):
                # Get the path to images
                list_img_dirs.append(os.path.join(scene_path, TEST_IMGS_SUBDIR))
                # Get the path to the texture
                for texture_name in os.listdir(scene_path):
                    if texture_name.lower().endswith(('.jpg', '.png')):
                        list_gt_textures.append(os.path.join(scene_path, texture_name))
                        break
    psnr_values = []
    ssim_values = []
    for imgs_dir_path, gt_texture_path in zip(list_img_dirs, list_gt_textures):
        print(f"Processing images from {imgs_dir_path}")    
        if(gt_texture_path):
            print(f"Texture: {gt_texture_path}")
        # Enhance the image using the MVTRN model
        image = enhance_image_multiview(model, imgs_dir_path, args.output_path,
            args.num_views, args.patch_size, args.patch_stride,
            args.max_image_size, num_workers=args.num_workers,
            alpha_threshold=args.alpha_threshold, device=device)

        # Normalize the image from np.float32 to np.uint8
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

        # If a path to a texture for comparison is given, compare image by metrics 
        if(gt_texture_path):
            gt_texture = load_image(gt_texture_path, args.max_image_size)
            print("GT texture shape:", gt_texture.shape, gt_texture.dtype)
            print("Enhanced output shape:", image.shape, image.dtype)
            
            psnr_value = compare_images(gt_texture, image, metric='PSNR')
            psnr_values.append(psnr_value)
            print(f"PSNR value: {psnr_value:.5f} db")
            ssim_value = compare_images(gt_texture, image, metric='SSIM')
            ssim_values.append(ssim_value)
            print(f"SSIM value: {ssim_value:.5f}")

        # If the script is run as reconstruction (not testing), plot output
        if(args.imgs_path):
            # Take the first view image file as the reference view and save too
            file_name = sorted([f for f in os.listdir(imgs_dir_path) if f.endswith(('.png', '.jpg', '.jpeg'))])[0]
            save_lr_ref_view_resized(
                image.shape[:2],
                os.path.join(imgs_dir_path, file_name),
                os.path.join(os.path.dirname(args.output_path), "ref_view.jpg"))
    
    # Print average total stats
    if(len(psnr_values) > 1):
        print(f"Average PSNR value: {sum(psnr_values) / len(psnr_values):.5f} db")
        print(f"Average SSIM value: {sum(ssim_values) / len(ssim_values):.5f}")
