import sys
sys.path.append('SEA-RAFT')
sys.path.append('SEA-RAFT/core')

import os
import cv2
import psutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from raft import RAFT

from image_utils import load_exr_to_array, find_bounding_box,\
    crop_image_to_bbox, resize_image, save_img


@torch.no_grad()
def align_image_hg_of(model, args, ref_view_img, view_img_path, uv_img_path, mask_object=True):
    # Cropped (and masked) view_img
    view_img = get_texture_area_image(view_img_path, uv_img_path, mask_object=mask_object)

    # Homography hard alignment
    # Align view_img to ref_view_img by homography transformation
    try:
        view_img = align_image_homography(ref_view_img, view_img)
    except:
        # Use the reference image instead of the view, when we cannot estimate
        # the homography transformation (low detail image)
        view_img = ref_view_img
        print("A transformation homography matrix was not found, the reference view was used")

    # Optical flow soft alignment
    view_img = align_image_opticalflow(model, args, ref_view_img, view_img)

    # Clip a float image and convert to uint8
    view_img = np.clip(view_img, 0, 255).astype(np.uint8)
    
    return view_img


def align_image_homography(ref_view_img, view_img):
    """Aligns the view image using homography transformation"""
    # Convert images to grayscale
    gray_ref = cv2.cvtColor(ref_view_img, cv2.COLOR_BGR2GRAY)
    gray_view = cv2.cvtColor(view_img, cv2.COLOR_BGR2GRAY)

    # Detect SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints_ref, descriptors_ref = sift.detectAndCompute(gray_ref, None)
    keypoints_view, descriptors_view = sift.detectAndCompute(gray_view, None)

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
def align_image_opticalflow(model, args, ref_view_img, view_img):
    """Aligns the view image to the reference using optical flow"""
    # Remove the alpha channels due to a 3 channel image SEA-RAFT model
    ref_view_img = cv2.cvtColor(ref_view_img, cv2.COLOR_RGBA2RGB)
    # Extract the alpha channel from view_img (the 4th channel)
    alpha_channel = view_img[:, :, -1] if view_img.shape[2] == 4 else None
    view_img = cv2.cvtColor(view_img, cv2.COLOR_RGBA2RGB)

    ref_view_img = torch.tensor(ref_view_img, dtype=torch.float32).permute(2, 0, 1)
    view_img = torch.tensor(view_img, dtype=torch.float32).permute(2, 0, 1)

    ref_view_img = ref_view_img[None].to(args.device)
    view_img = view_img[None].to(args.device)

    original_size = ref_view_img.shape[2:] # Tensor [B, C, H, W]
    scale_factor = 1
    ref_view_img_resized = ref_view_img
    view_img_resized = view_img
    # Resize before flow calculation if args.img_max_size is set
    if(args.img_max_size > 0):
        ref_view_img_resized, scale_factor = resize_image_to_max_side(ref_view_img, max_size=args.img_max_size)
        view_img_resized, _ = resize_image_to_max_side(view_img, max_size=args.img_max_size)

    # Calculate the forward optical flow (flow_up) on RGB images
    flow = model(ref_view_img_resized, view_img_resized, iters=args.iters, test_mode=True)['flow'][-1]
    if(args.img_max_size > 0):
        # Scale the flow back to the original resolution
        flow = F.interpolate(flow, size=original_size, mode='bilinear',
                                align_corners=False) * (1.0 / scale_factor)
    
    if alpha_channel is not None:
        # Add the alpha channel to an image tensor befor applying the flow
        alpha_channel = torch.tensor(alpha_channel, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(args.device)
        view_img = torch.cat((view_img, alpha_channel), dim=1)  # [B, 4, H, W]

    aligned_img = align_flow_to_image(view_img, flow)

    # Convert back to (H, W, C) numpy array and normalize
    aligned_img = aligned_img[0].cpu().detach().permute(1, 2, 0).numpy()

    return aligned_img


class Settings:
    """Model settings and arguments"""
    def __init__(self, iterations=4, img_max_size=1500,
                    model="MemorySlices/Tartan-C-T-TSKH-kitti432x960-M"):
        # Set const attributes from the JSON config (e.g."kitti-M.json")
        self.dim = 128
        self.radius = 4
        self.block_dims = [64, 128, 256]
        self.initial_dim = 64
        self.pretrain = "resnet34"
        self.iters = iterations
        self.num_blocks = 2

        # Additional attributes required for SEA-RAFT
        self.path = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.url = model # Pretrained model HuggingFace URL
        # Maximal model input size of the image (flow will be upscaled back)
        self.img_max_size = img_max_size # -1 to not resize

        
def init_searaft(settings):
    """Initialize settings and the optical flow SEA-RAFT model"""
    
    args = settings # Settings

    model = RAFT.from_pretrained(args.url, args=args)
    model = model.to(args.device)
    model.eval()

    return model, args


def get_texture_area_image(view_img_path, uv_img_path, mask_object=True):
    """Crop and mask the texture area of the image view"""
    uv_img = load_exr_to_array(uv_img_path)
    alpha_channel = uv_img[:, :, -1]
    view_img = cv2.imread(view_img_path, cv2.IMREAD_UNCHANGED)
    bbox = find_bounding_box(alpha_channel)

    # Crop just the surface object using UV mask rectangle
    alpha_channel_cropped = crop_image_to_bbox(alpha_channel, bbox)
    view_img_cropped = crop_image_to_bbox(view_img, bbox)

    # Extract (mask) the surface visible area
    if(mask_object == True):
        alpha_mask = (alpha_channel_cropped > 0).astype(np.uint8) * 255
        view_img_masked = cv2.bitwise_and(view_img_cropped, view_img_cropped, mask=alpha_mask)
        return view_img_masked
    return view_img_cropped


def get_ref_image(texture_shape, ref_view_img_path, ref_uv_img_path, mask_object=True):
    """Crop and mask the texture area from the reference image"""
    # Get the reference texture dimensions
    texture_h, texture_w, _ = texture_shape
    max_size = max(texture_h, texture_w)

    ref_view_img_masked = get_texture_area_image(ref_view_img_path, ref_uv_img_path, mask_object)

    # The output reference texture view is resized to the known texture shape
    # In practice there would be probably constant max resolution to resize to
    ref_view_img_resized = resize_image(ref_view_img_masked, max_size)
    return ref_view_img_resized


def resize_image_to_max_side(image, max_size=1000):
    """Resize an image so that is's longer side is not bigger than max_size"""
    H, W = image.shape[2:]
    scale_factor = max_size / max(H, W)
    new_size = (int(H * scale_factor), int(W * scale_factor))
    image = F.interpolate(image, size=new_size, mode='bilinear', align_corners=False)
    return image, scale_factor


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
