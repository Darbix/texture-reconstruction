# Scripts for image alignment of views
## Method 1: UV mapping views to textures

Example for UV mapping:
```
python main.py --data_path "camera_images" --texture_path "" --output_path "aligned_data" --mask --uv_upscale 1.5
```

## Method 2: Homography with Optical flow alignment of views to reference ones
Example for Homography with Optical flow (SEA-RAFT model):
```
python main.py --data_path "camera_images" --texture_path "" --output_path "aligned_data" --hgof --mask --iters 6 --max_size 2000
```

## References

- **SEA-RAFT**: Simple, Efficient, Accurate RAFT for Optical Flow [GitHub](https://github.com/princeton-vl/SEA-RAFT) [Paper](https://arxiv.org/abs/2405.14793)
