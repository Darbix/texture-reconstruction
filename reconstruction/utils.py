# utlis.py

import os
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch



def split_list(data, n):
    # Calculate the size of each part
    part_size = len(data) // n
    remainder = len(data) % n

    parts = []
    start = 0

    for i in range(n):
        # Calculate the end index for the current part
        end = start + part_size + (1 if i < remainder else 0)
        # Append the sublist to the parts list
        parts.append(data[start:end])
        # Update the start index for the next part
        start = end

    return parts


# Needed for all methods of loading
def get_crop_positions(idx, height, width, tile_size, tile_stride):
    """Generate all possible crop positions (tiles) within an image."""
    crop_positions = []

    padded_height = math.ceil(height / tile_size) * tile_size
    padded_width = math.ceil(width / tile_size) * tile_size

    # Generate all tile positions for cropping
    for i in range(0, padded_height + 1 - tile_size, tile_stride):
        for j in range(0, padded_width + 1 - tile_size, tile_stride):
            crop_positions.append((idx, i, j))

    return crop_positions, padded_height, padded_width


# Needed for all methods of loading
def get_data_for_tiles(scene_paths, tile_size, tile_stride, input_max_res):
    """Load data for particular tiles"""
    scene_tile_indices = []
    scene_padded_heights = []
    scene_padded_widths = []
    scene_tile_counts = []

    for idx, scene_path in enumerate(scene_paths):
        # Use the first view as input image size reference for tiles and padding
        first_image_file = [
            f for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))
        ][0]
        image_path = os.path.join(scene_path, first_image_file)

        with Image.open(image_path) as img:
            img_w, img_h = img.size[:2]
            if (input_max_res is not None):
                scale = input_max_res / max(img_w, img_h)
                img_w, img_h = int(img_w * scale), int(img_h * scale)

            crop_positions, pH, pW = get_crop_positions(idx, img_h, img_w, tile_size, tile_stride)
            scene_tile_indices += crop_positions
            scene_tile_counts += [len(crop_positions)]
            scene_padded_heights += [pH]
            scene_padded_widths += [pW]

    return scene_tile_indices, scene_padded_heights, scene_padded_widths, scene_tile_counts


def crop_tile(img, tile_x, tile_y, tile_size, transform_tile):
    """Extract a tile from a cached full-size image."""
    tile = img.crop((tile_x, tile_y, tile_x + tile_size, tile_y + tile_size))
    return transform_tile(tile)


def attach_tiles_to_images(
        composed_image, img_size=None, sr_factor=1,
        tile=None, tile_x=None, tile_y=None):
    """Adds a tile from the image to the final one at the specified location"""
    tile_size = tile.shape[1] # H == W so the tile_size is any of that

    if(sr_factor > 1):
        tile_x = int(tile_x * sr_factor)
        tile_y = int(tile_y * sr_factor)
        img_size = (img_size[0],                  # C 
                    int(img_size[1] * sr_factor), # H
                    int(img_size[2] * sr_factor)) # W

    # Init images if it's the first tile
    if(composed_image is None):
        # Init images in original shapes
        composed_image = np.zeros((img_size[1], img_size[2], img_size[0]))

    # Add a current tile to the composed SR image
    tile = tile.cpu().detach().numpy().transpose(1, 2, 0)
    # rnd_sr_img_tile = b_sr_img_tile[sample_n].cpu().detach().numpy().transpose(1, 2, 0)
    composed_image[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size, :] = tile

    return composed_image




# # TODO export batch, image sizes and process 1 image, not 3
# def attach_tiles_to_images(
#         rnd_lr_img, rnd_sr_img, rnd_hr_img, img_size=None,
#         b_lr_imgs=None, tile_size=None, b_sr_img_tile=None,
#         b_hr_texture=None, tile_x=None, tile_y=None,
#         sample_n=0):
#     """
#     Adds a tile from the LR and SR batch images to the final ones
#     at the specified location and also selects the corresponding HR texture.

#     Parameters:
#     - rnd_lr_img: Current LR image to update.
#     - rnd_sr_img: Current SR image to update.
#     - rnd_hr_img: Current HR image with the selected nth scene target texture.
#     - all_indices: Tile indices (pairs) for the batch.
#     - b_lr_imgs: Batch of LR tiles.
#     - tile_size: Size of a tile.
#     - b_sr_img_tile: Batch of SR tiles (crops at a single tile position).
#     - b_hr_texture: Batch of HR textures.
#     - tile_x, tile_y: Position of the tile.
#     - sample_n: Index for the sample from the batch to select.

#     Returns:
#     - Updated rnd_lr_img, rnd_sr_img, and rnd_hr_img.
#     """

#     # Compute the full input image size based on tile positions and tile size
#     input_img_size = (b_lr_imgs.shape[2], img_size[0], img_size[1]) # [C, H, W]
#     # Compute the SR model scaling factor
#     sr_factor = b_sr_img_tile.shape[3] / b_lr_imgs.shape[4]
#     # Get the real SR output image size using the enlargement factor
#     sr_img_size = (b_sr_img_tile.shape[1],              # C
#                     int(input_img_size[1] * sr_factor), # H
#                     int(input_img_size[2] * sr_factor)) # W

#     # Init images if it's the first tile
#     if(rnd_sr_img is None and rnd_lr_img is None):
#         # Init images in original shapes
#         rnd_sr_img = np.zeros((sr_img_size[1], sr_img_size[2], sr_img_size[0]))
#         rnd_lr_img = np.zeros((input_img_size[1], input_img_size[2], input_img_size[0]))
#         rnd_hr_img = np.zeros((sr_img_size[1], sr_img_size[2], sr_img_size[0]))

#     # Add a current tile to the composed SR image
#     rnd_sr_img_tile = b_sr_img_tile[sample_n].cpu().detach().numpy().transpose(1, 2, 0)
#     rnd_sr_img[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size, :] = rnd_sr_img_tile

#     # Add a current tile to the composed HR image
#     rnd_hr_img_tile = b_hr_texture[sample_n].cpu().detach().numpy().transpose(1, 2, 0)
#     rnd_hr_img[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size, :] = rnd_hr_img_tile

#     # Add a current tile (of a first view) to the composed LR image
#     rnd_lr_img_tile = b_lr_imgs[sample_n][0].cpu().detach().numpy().transpose(1, 2, 0)
#     rnd_lr_img[tile_y:tile_y + tile_size, tile_x:tile_x + tile_size, :] = rnd_lr_img_tile

#     return rnd_lr_img, rnd_sr_img, rnd_hr_img




def plot_epoch_images(lr_img, sr_img, hr_img, output_file_path, path_individuals=""):
    """Plot first MVTRN input reference view, output texture and target one"""

    imgs = [lr_img, sr_img, hr_img]
    titles = ['Low-Resolution reference view',
              'Super-Resolved output',
              'Ground-Truth texture']

    file_name = ["plot_LR.png", "plot_SR.png", "plot_HR.png"]

    plt.figure(figsize=(12, 4))

    for i, (img, title) in enumerate(zip(imgs, titles)):
        plt.subplot(1, 3, i+1)
        image = np.clip((img + 1) / 2, 0, 1)
        plt.imshow(image)
        plt.title(title)
        plt.axis('off')

        if path_individuals:
            plt.imsave(os.path.join(path_individuals, file_name[i]), image)

    if output_file_path:
        plt.savefig(output_file_path, dpi=600, bbox_inches='tight')

    # TODO only Colab
    plt.show()
    plt.close()

    return


# def get_transform(resolution=None):
#     """Creates a composed data transformation"""
#     # Convert to square shape
#     transforms = [PaddingToSquare(fill=(0, 0, 0, 0))]
#     # Resize if required
#     if resolution is not None:
#         transforms.append(T.Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC))
#     transforms.extend([
#         T.ToTensor(),
#         # Normalize to [-1, 1]
#         T.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),
#     ])

#     return T.Compose(transforms)


def resize_max_side(image, max_size):
    """Lambda transform to resize the longer side and keep the aspect ratio"""
    w, h = image.size
    scale = max_size / max(w, h)  # Scale factor for the longer side
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.BICUBIC)







# Uses a cache for loading a test sample in cropped tiles
def load_data(views_path, texture_path, transform_view=None, transform_texture=None,
              transform_tile=None, n=-1, tile_size=256, input_max_res=None):

    tile_stride = tile_size
    td = get_data_for_tiles([views_path], tile_size, tile_stride, input_max_res)
    scene_tile_indices, scene_padded_heights, scene_padded_widths, scene_tile_counts = td

    # Load full images once
    texture_img = load_full_texture(texture_path, transform_texture)
    view_imgs = load_full_views(views_path, transform_view, n)

    list_tiles = []

    for idx, tile_y, tile_x in scene_tile_indices:
        ph, pw = scene_padded_heights[idx], scene_padded_widths[idx]

        # Crop from cached images
        texture_tile = crop_tile(texture_img, tile_x, tile_y, tile_size, transform_tile)
        view_tiles = [crop_tile(img, tile_x, tile_y, tile_size, transform_tile) for img in view_imgs]
        view_tiles = torch.stack(view_tiles)

        # Tile info as: (scene_idx, tile_y, tile_x, tc, ph, pw)
        tile = torch.tensor((idx, tile_y, tile_x,
                             scene_tile_counts[idx], ph, pw))

        list_tiles.append((view_tiles, texture_tile, tile))

    return list_tiles


def load_full_texture(texture_path, transform_texture):
    """Load full texture image into memory."""
    with Image.open(texture_path) as img:
        img = img.convert("RGBA")
        return transform_texture(img)


def load_full_views(views_dir, transform_view, n):
    """Load all view images into memory."""
    view_files = sorted([f for f in os.listdir(views_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:n]

    view_imgs = []
    for view_file in view_files:
        view_path = os.path.join(views_dir, view_file)
        with Image.open(view_path) as img:
            img = img.convert("RGBA")
            view_imgs.append(transform_view(img))

    return view_imgs
