# Scripts for image alignment of views
## Dataset image alignment (`get_aligned_dataset.py`)
### Method 1: UV mapping views to textures
Example for UV mapping:
```
python main.py --data_path "camera_images" --texture_path "" --output_path "aligned_data" --mask --uv_upscale 1.5 --compression 9
```

### Method 2: Homography with Optical flow alignment of views to reference ones
Example for Homography with Optical flow (SEA-RAFT model):
```
python get_aligned_dataset.py --data_path "camera_images" --texture_path "texture_images" --output_path "aligned_data_hgof" --hgof --mask --max_size_hg 1024 --max_size_of 1024 --patch_size 3072 --patch_stride 2048  --model_name "sea_raft_m" --ckpt_path "kitti"
```
Which expects a data structure: `camera_images/<scene_N_name>` with `data.txt` (texture name inside) and `color_imgs` with view images in each `camera_images/<scene_N_name>`. The texture name is a path to `texture_images/` with textures. Homography is calculated on images with a max size 1024x1024, the optical flow processes the images in tiles of 3072x3072 size with stride 2048 and resizes them into 1024x1024 before fetching to a specific optical flow model given by `--model_name` and `ckpt_path`.

## Photo alignment (`align_photos.py`)
Aligns real photos in a directory given by `--scene_path` to a reference view (first image file in an alphabetical order) resized to `--max_image_size` max size.
Example usage:
```
python data_alignment/align_photos.py --scene_path "path_to_images" --output_path "path_to_output_dir" --max_size_hg 1024 --max_size_of 512 --patch_size 3072 --patch_stride 2048 --max_image_size 8000 --model_name "sea_raft_s" --ckpt_path "kitti"
```

## References
- **PTLFlow**: PyTorch Lightning Optical Flow [GitHub](https://github.com/hmorimitsu/ptlflow)
- **SEA-RAFT**: Simple, Efficient, Accurate RAFT for Optical Flow [GitHub](https://github.com/princeton-vl/SEA-RAFT) [Paper](https://arxiv.org/abs/2405.14793)
