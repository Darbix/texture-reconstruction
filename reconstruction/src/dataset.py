# dataset.py

import os
import re
import json
import bisect
import random
from PIL import Image, ImageFilter
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from utils import load_patch_set, string_to_position
import config


class MultiViewDataset(Dataset):
    """Dataset loader with worker caching and parallelization"""
    def __init__(self, data_path,
                 transform_patch=None,
                 n_patches=-1, input_max_res=None,
                 split_ratio=0.8, train=True):

        # Other initialization
        self.n_patches = n_patches
        # self.s = s if s is not None else -1 # Debug limit the number of scenes
        self.data_path = data_path
        self.transform_patch = transform_patch
        self.input_max_res = input_max_res

        self.scene_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        # self.scene_dirs = self.scene_dirs[:self.s] if self.s > 0 else self.scene_dirs
        # Dataset split
        split_index = int(len(self.scene_dirs) * split_ratio)
        self.scene_dirs = self.scene_dirs[:split_index] if train else self.scene_dirs[split_index:]
        self.scene_paths = [os.path.join(data_path, d) for d in self.scene_dirs]

        # Accumulated number of texture patches per each scene
        self.acc_patch_indices = []     # List of accumulated number of scene texture patches
        self.scene_patch_positions = [] # List of lists of texture patch positions
        # Data as (scene_index, patch_y, patch_x, texture_h, texture_w, patch_size)
        self.scene_data = [] # Image and patch information per scene
        patch_idx = 0

        # For each scene get info about patches and append data to lists
        for scene_path in self.scene_paths:
            texture_patch_dir = os.path.join(scene_path, config.TEXTURE_PATCHES_DIR)
            
            if os.path.exists(texture_patch_dir) and os.path.isdir(texture_patch_dir):
                # Count files (patches) in a directory of texture patches
                texture_patches = [f for f in os.listdir(texture_patch_dir) if os.path.isfile(os.path.join(texture_patch_dir, f))]
                patch_idx += len(texture_patches)
                self.acc_patch_indices.append(patch_idx)
                
                # Extract unique patch positions
                patch_positions = []
                for filename in texture_patches:
                    y, x = string_to_position(filename)
                    patch_positions.append((y, x))
                self.scene_patch_positions.append(patch_positions)

            # Extract useful scene information about images and patches
            info_path = os.path.join(scene_path, "info.json")
            with open(info_path, 'r') as file:
                data = json.load(file)
            self.scene_data.append((data["texture_h"], data["texture_w"], data["patch_size"]))

    def __getitem__(self, idx):
        scene_idx, patch_idx = self.get_indices(idx)
        patch_y, patch_x = self.scene_patch_positions[scene_idx][patch_idx]
        patch_views, patch_texture = self.get_patches(scene_idx, patch_y, patch_x)

        patch_info = torch.tensor((
            scene_idx, patch_y, patch_x,   # I, PY, PX
            self.scene_data[scene_idx][0], # texture_h (TH)
            self.scene_data[scene_idx][1], # texture_w (TW)
            self.scene_data[scene_idx][2]  # patch_size (PS)
            ))
        
        # [N, C, H, W], [C, H, W], [I, PY, PX, TH, TW, PS]
        return patch_views, patch_texture, patch_info

    def get_patches(self, scene_idx, patch_y, patch_x):
        scene_path = self.scene_paths[scene_idx]
        texture_patches_dir_path = os.path.join(scene_path, config.TEXTURE_PATCHES_DIR)
        view_patches_dir_path =    os.path.join(scene_path, config.VIEW_PATCHES_DIR)
        ref_patches_dir_path =     os.path.join(scene_path, config.REF_PATCHES_DIR)
        
        # For debug only, the patch_size won't be resized in practice
        patch_size = self.input_max_res if self.input_max_res > 0 else self.scene_data[scene_idx][2]

        texture_patch, view_patches = load_patch_set(
            (patch_y, patch_x),
            texture_patches_dir_path, 
            ref_patches_dir_path, 
            view_patches_dir_path,
            self.transform_patch, patch_size, self.n_patches, 
            max_patch_size=patch_size)

        view_patches = torch.stack(view_patches)
        return view_patches, texture_patch

    def __len__(self):
        return self.acc_patch_indices[-1]

    def get_indices(self, idx):
        """Get the scene index from the index of the patch using accumulated counts of patches"""
        scene_idx = bisect.bisect_right(self.acc_patch_indices, idx) 
        patch_idx = self.acc_patch_indices[scene_idx - 1] - idx if scene_idx > 0 else idx
        return scene_idx, patch_idx
