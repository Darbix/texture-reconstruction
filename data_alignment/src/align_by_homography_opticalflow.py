# align_by_homography_opticalflow.py

import os
import cv2
import math
import psutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import ptlflow
from ptlflow.utils import flow_utils
from ptlflow.utils.io_adapter import IOAdapter
from concurrent.futures import ThreadPoolExecutor

from image_utils import load_exr_to_array, find_bounding_box,\
    crop_image_to_bbox, resize_to_max_size, save_img


def align_image_hg_of_with_tiling(model, args, ref_view_img, view_img_path,
    uv_img_path, mask_object=True, max_size_hg=-1, max_size_of=-1,
    patch_size=2048, patch_stride=1536):
    """
    Aligns the view image to the reference view image using
    homography and optical flow. Optical flow is done using tiling method.
    """
    # Read and process the view image and UV map (cropped based on UV mask)
    view_img = get_texture_area_image(view_img_path, uv_img_path, mask_object=mask_object)

    # Homography hard alignment
    try:
        view_img = align_image_homography(ref_view_img, view_img, max_size=max_size_hg)
    except:
        view_img = ref_view_img
        print("A transformation homography matrix was not found, the reference view was used")

    # Optical flow soft alignment with tiling
    aligned_img = align_image_opticalflow_tiling(
        ref_view_img, view_img, model, max_size=max_size_of,
        patch_size=patch_size, patch_stride=patch_stride)
    
    # Clip to uint8 and return the result
    aligned_img = np.clip(aligned_img, 0, 255).astype(np.uint8)

    return aligned_img


def get_patch_weight_blend_mask(patch_size):
    """Creates a mask of weights for gradient blend of patches"""
    # Auxiliary variables to create a gradient mask
    y_grid = np.linspace(0, 2, patch_size).reshape(-1, 1)
    x_grid = np.linspace(0, 2, patch_size).reshape(1, -1)
    # Mask in range of values (0.0 - 1.0) with higher weight in the center
    blend_mask = np.minimum(np.minimum(y_grid, 2 - y_grid),
                            np.minimum(x_grid, 2 - x_grid))
    blend_mask = np.expand_dims(blend_mask, axis=-1) # [H, W, 1]
    
    # To strengthen the patch center weight over the edges
    blend_mask = blend_mask ** 3

    blend_mask = np.clip(blend_mask, 1e-7, None)

    return blend_mask


def normalize_composed_image(composed_image, weight_map):
    """Normalizes the image to [0.0, 1.0] using the accumulated weight map"""
    # Avoid a division by zero
    weight_map = np.clip(weight_map, 1e-7, None)
    return (composed_image / weight_map)


def process_patch(y, x, ref_img_padded, tgt_img_padded, model, patch_size,
    max_size):
    """Processes a single tile and returns its contribution"""
    y_end = y + patch_size
    x_end = x + patch_size

    ref_patch = ref_img_padded[y:y_end, x:x_end]
    tgt_patch = tgt_img_padded[y:y_end, x:x_end]

    # Skip processing if the patch has all values zeros
    if np.all(tgt_patch < 1e-6):
        print(f"Patch y: {y}, x: {x} skipped due to zero values") 
        return (y, x, np.zeros_like(ref_patch))

    # Apply the optical flow model on a patch
    aligned_patch = align_image_opticalflow(
        ref_patch, tgt_patch, model, max_size=max_size
    )

    return (y, x, aligned_patch)


def align_image_opticalflow_tiling(ref_img, tgt_img, model, max_size=-1,
    patch_size=2048, patch_stride=1536):
    # Calculate the image size with padding
    img_height, img_width = ref_img.shape[:2]
    padded_img_height = math.ceil((img_height - patch_size) / patch_stride) * patch_stride + patch_size
    padded_img_width = math.ceil((img_width - patch_size) / patch_stride) * patch_stride + patch_size
    # Pad the reference and view images and normalize
    ref_img_padded = np.pad(ref_img[:, :, :3],
        ((0, padded_img_height - img_height), (0, padded_img_width - img_width),
        (0, 0)), mode='constant') / 255
    tgt_img_padded = np.pad(tgt_img[:, :, :3],
        ((0, padded_img_height - img_height), (0, padded_img_width - img_width),
        (0, 0)), mode='constant') / 255

    # Initalize an image to compose patches, a mask for blending and pixel weights
    aligned_img = np.zeros_like(tgt_img_padded, dtype=np.float32)
    blend_mask = get_patch_weight_blend_mask(patch_size)
    weight_map = np.zeros((tgt_img_padded.shape[0], tgt_img_padded.shape[1], 1),
        dtype=np.float32)

    tasks = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for y in range(0, padded_img_height - patch_size + 1, patch_stride):
            for x in range(0, padded_img_width - patch_size + 1, patch_stride):
                tasks.append(executor.submit(process_patch, y, x, 
                    ref_img_padded, tgt_img_padded, model, patch_size, max_size))

        for future in tasks:
            y, x, aligned_tile = future.result()
            y_end = y + patch_size
            x_end = x + patch_size
            # Attach a patch to the composed image
            aligned_img[y:y_end, x:x_end, :] += aligned_tile * blend_mask
            # Add a blend mask to total sum of weights
            weight_map[y:y_end, x:x_end, :] += blend_mask

    # Normalize the output back to 0.0-1.0 using weights
    aligned_img = normalize_composed_image(aligned_img, weight_map)
    # Crop the image area without padding and convert to 0-255
    aligned_img_cropped = aligned_img[:img_height, :img_width] * 255
    # Restore the alpha channel if exists
    if tgt_img.shape[2] == 4:
        aligned_img_cropped = np.dstack((aligned_img_cropped, tgt_img[:, :, 3]))

    return aligned_img_cropped


def align_image_homography(ref_view_img, view_img, max_size=-1):
    """Aligns the view image using homography transformation"""
    # Resize image to speed up the process and convert to grayscale
    gray_ref = resize_to_max_size(
        cv2.cvtColor(ref_view_img, cv2.COLOR_BGR2GRAY), max_size=max_size)
    gray_view = resize_to_max_size(
        cv2.cvtColor(view_img, cv2.COLOR_BGR2GRAY), max_size=max_size)

    # Detect SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_view, descriptors_view = sift.detectAndCompute(gray_view, None)

    # Get the keypoints in the original image sizes
    scale_back_ref = ref_view_img.shape[1] / gray_ref.shape[1]
    scale_back_view = view_img.shape[1] / gray_view.shape[1]

    keypoints_ref = [
        cv2.KeyPoint(
            kp.pt[0] * scale_back_ref,
            kp.pt[1] * scale_back_ref,
            kp.size * scale_back_ref) for kp in keypoints_ref
    ]
    keypoints_view = [
        cv2.KeyPoint(
            kp.pt[0] * scale_back_view,
            kp.pt[1] * scale_back_view,
            kp.size * scale_back_view) for kp in keypoints_view
    ]

    # Ensure descriptors are not empty
    if descriptors_ref is None or descriptors_view is None:
        raise ValueError("SIFT could not find enough features in one of the images")

    # FLANN-based matcher parameters
    index_params = dict(algorithm=1, trees=8) # FLANN_INDEX_KDTREE
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match features using KNN
    matches = flann.knnMatch(descriptors_ref, descriptors_view, k=2)

    # Apply Lowe’s ratio test
    threshold = 0.75
    tries = 2
    for i in range(tries):
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        # Ensure enough matches are found
        if(len(good_matches) >= 4):
            break
        elif i < tries:
            threshold += 0.05
        else:
            raise ValueError("Not enough good matches to compute homography")

    # Extract matched keypoints
    ref_pts = np.float32([keypoints_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    view_pts = np.float32([keypoints_view[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC
    H, mask = cv2.findHomography(view_pts, ref_pts, cv2.RANSAC, 5.0)

    if H is None:
        raise ValueError("Homography computation failed")

    # Warp the view to align with the reference image
    h, w, _ = ref_view_img.shape
    aligned_img = cv2.warpPerspective(view_img, H, (w, h))

    return aligned_img


@torch.no_grad()
def align_image_opticalflow(ref_img, tgt_img, model, max_size=-1):
    """Aligns the target image using optical flow transformation"""

    # Resize ref and target images to max_size for faster flow calculation
    ref_img_resized = resize_to_max_size(ref_img, max_size=max_size)
    tgt_img_resized = resize_to_max_size(tgt_img, max_size=max_size)

    # Prepare inputs for the model
    io_adapter = IOAdapter(model, ref_img_resized.shape[:2])
    inputs = io_adapter.prepare_inputs([ref_img_resized, tgt_img_resized])

    # Predict optical flow
    with torch.no_grad():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        inputs = {key: value.to(device) for key, value in inputs.items()}

        predictions = model(inputs)
        flow = predictions['flows'][0][0]  # Extract a flow tensor (2, H, W)

    # Convert flow to numpy
    flow_np = flow.permute(1, 2, 0).cpu().numpy()

    # Resize flow to match the original image size
    h, w = ref_img.shape[:2]  # Get the original size of the reference image
    flow_resized = cv2.resize(flow_np, (w, h), interpolation=cv2.INTER_LINEAR)

    # Generate meshgrid of pixel coordinates (for original-sized image)
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # Map flow to the original image coordinates
    map_x = (x + flow_resized[..., 0]).astype(np.float32)
    map_y = (y + flow_resized[..., 1]).astype(np.float32)

    # Warp target image using resized flow (apply to original size)
    aligned_img = cv2.remap(tgt_img, map_x, map_y, cv2.INTER_LINEAR)

    return aligned_img


def init_opticalflow_model(model_name='sea_raft_s', ckpt_path='kitti'):
    """Initialize an opticalflow model with pretrained weights"""
    model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)
    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


def get_texture_area_image(view_img_path, uv_img_path, mask_object=True):
    """Crop and mask the texture area of the image view"""
    view_img = cv2.imread(view_img_path, cv2.IMREAD_UNCHANGED)
    
    if not uv_img_path:
       return view_img
    
    uv_img = load_exr_to_array(uv_img_path)
    alpha_channel = uv_img[:, :, -1]
    bbox = find_bounding_box(alpha_channel)

    # Crop just the surface object using UV mask rectangle
    alpha_channel_cropped = crop_image_to_bbox(alpha_channel, bbox)
    view_img_cropped = crop_image_to_bbox(view_img, bbox)

    # Extract (mask) the surface visible area
    if(mask_object == True):
        alpha_mask = (alpha_channel_cropped > 0).astype(np.uint8) * 255
        view_img_masked = cv2.bitwise_and(view_img_cropped, view_img_cropped,
            mask=alpha_mask)
        return view_img_masked
    return view_img_cropped


def get_ref_image(max_size, ref_view_img_path, uv_path=None,
    mask_object=False):
    """Crop and mask the texture area from the reference image"""
    ref_view_img_masked = get_texture_area_image(
        ref_view_img_path, uv_path, mask_object)

    # The output reference texture view is resized to the specific max size
    ref_view_img_resized = resize_to_max_size(
        ref_view_img_masked, max_size=max_size)
    return ref_view_img_resized


def align_flow_to_image(image, flow):
    """Warp the image using the given optical flow"""
    B, C, H, W = image.shape

    # Create mesh grid of coordinates
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=0).float().to(image.device) # (2, H, W)

    # Normalize grid to [-1, 1] range for grid_sample
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)
    flow_grid = grid + flow

    # Normalize flow_grid to range [-1, 1]
    flow_grid[:, 0, :, :] = (flow_grid[:, 0, :, :] / (W - 1)) * 2 - 1  # X coords
    flow_grid[:, 1, :, :] = (flow_grid[:, 1, :, :] / (H - 1)) * 2 - 1  # Y coords

    # Rearrange for grid_sample: (B, H, W, 2)
    flow_grid = flow_grid.permute(0, 2, 3, 1)

    # Warp image using bilinear interpolation
    warped_image = F.grid_sample(image, flow_grid, mode='bilinear', align_corners=True)

    return warped_image


def print_ram():
    """Prints free RAM memory"""
    print(f"Available RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")
