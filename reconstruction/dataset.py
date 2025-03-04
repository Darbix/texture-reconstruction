# dataset.py

import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset, get_worker_info

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Lock

from utils import get_data_for_tiles, split_list



class MultiViewDataset(Dataset):
    """Dataset loader with worker caching and parallelization"""
    def __init__(self, data_path, transform_view=None, transform_texture=None,
                 transform_tile=None, tile_size=256, tile_stride=None,
                 n=-1, s=-1, input_max_res=None, num_workers=1,
                 max_workers_loading=1, split_ratio=0.8, train=True):

        # Only one scene is loaded per worker
        self.worker_cache = {}

        # Other initialization
        self.n = n
        self.s = s if s is not None else -1
        self.data_path = data_path
        self.transform_view = transform_view
        self.transform_texture = transform_texture
        self.transform_tile = transform_tile
        self.tile_size = tile_size
        self.tile_stride = tile_size if tile_stride is None else tile_stride
        self.input_max_res = input_max_res
        self.num_workers = num_workers
        self.max_workers_loading = max_workers_loading

        self.scene_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
        self.scene_dirs = self.scene_dirs[:self.s] if self.s > 0 else self.scene_dirs
        # ----- Dataset split -----
        random.shuffle(self.scene_dirs)
        split_index = int(len(self.scene_dirs) * split_ratio)
        if train:
            self.scene_dirs = self.scene_dirs[:split_index]
        else:
            self.scene_dirs = self.scene_dirs[split_index:]
        # -------------------------
        self.scene_paths = [os.path.join(data_path, d) for d in self.scene_dirs]

        # Get complex data for all tiles
        td = get_data_for_tiles(self.scene_paths, self.tile_size, self.tile_stride, self.input_max_res)
        # self.scene_tile_indices: [T, (scene_index, y_pos, x_pos)]
        # self.scene_padded_<heights, widths>: [S] list of input image sizes padded to fit tiles
        # self.scene_tile_counts: [S] list of tile counts per scene
        self.scene_tile_indices, self.scene_padded_heights, self.scene_padded_widths, self.scene_tile_counts = td

        # Split to num_workers parts indexed by IDs of workers
        self.scene_tile_indices = split_list(self.scene_tile_indices, self.num_workers)
        # TODO
        print([len(p) for p in self.scene_tile_indices])

        self.print_lock = Lock()


    def load_scene_views(self, scene_idx, max_workers_loading):
        """Load and return all views and textures for a given scene index."""
        if scene_idx in self.worker_cache:
            return self.worker_cache[scene_idx]

        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0

        scene_path = self.scene_paths[scene_idx]

        texture_file = next(f for f in os.listdir(scene_path) if f.endswith(('.png', '.jpg', '.jpeg')))
        texture_path = os.path.join(scene_path, texture_file)
        with Image.open(texture_path) as img:
            img = img.convert("RGBA")
            texture = self.transform_texture(img)

        views_dir = os.path.join(scene_path, 'color_imgs')
        view_files = sorted([f for f in os.listdir(views_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])[:self.n]


        # Method 1
        # views = []
        # for view_file in view_files:
        #     view_path = os.path.join(views_dir, view_file)
        #     with Image.open(view_path) as img:
        #         img = img.convert("RGBA")
        #         views.append(self.transform_view(img))

        # Method 2
        # Parallelized loading views
        def load_and_transform_image(view_path, transform_view):
            with Image.open(view_path) as img:
                img = img.convert("RGBA")
                return transform_view(img)

        with ThreadPoolExecutor(max_workers=max_workers_loading) as executor:
            # Submit tasks to the executor
            futures = [executor.submit(load_and_transform_image,
                                       os.path.join(views_dir, view_file),
                                       self.transform_view)
                      for view_file in view_files]
            # Collect the results as they are completed
            views = [future.result() for future in futures]

        self.worker_cache.clear()
        self.worker_cache[scene_idx] = (views, texture)

        with self.print_lock:
            print(f"Loaded scene {scene_idx} from worker {worker_id}")

        return views, texture

    def __getitem__(self, idx):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        local_idx = (idx - worker_id) // self.num_workers

        # Get tile info from the worker-specific list index
        scene_idx, tile_y, tile_x = self.scene_tile_indices[worker_id][local_idx]

        # Load views and textures from worker cache
        views, texture = self.load_scene_views(scene_idx, self.max_workers_loading)

        # Crop texture tile
        texture_tile = texture.crop((tile_x, tile_y, tile_x + self.tile_size, tile_y + self.tile_size))
        texture_tile = self.transform_tile(texture_tile)

        # Crop view tiles
        view_tiles = [view.crop((tile_x, tile_y, tile_x + self.tile_size, tile_y + self.tile_size)) for view in views]
        view_tiles = [self.transform_tile(tile) for tile in view_tiles]
        view_tiles = torch.stack(view_tiles)

        tile = torch.tensor((scene_idx, tile_y, tile_x,
                             self.scene_tile_counts[scene_idx],
                             self.scene_padded_heights[scene_idx],
                             self.scene_padded_widths[scene_idx]))

        return view_tiles, texture_tile, tile

    def __len__(self):
        return sum(len(part) for part in self.scene_tile_indices)

    def num_of_scenes(self):
        return len(self.scene_dirs)
