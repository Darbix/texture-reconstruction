# Data tiling and realistic photo degradation
The script is used for cropping large scene image views into patches of a given size.
Example usage:
```
python extract_patches.py --data_dir data --output_dir output --patch_size 512
```
This script extracts patches from texture images within a scene directory in `--data_dir`. The process involves degrading the texture and using it as a reference view while degrading all images in the `color_imgs/` directory to create patches from them. By default, the script generates `texture_patches`, `reference_patches`, `image_patches` and `info.json` in each scene directory. Patches that are almost completely transparent are skipped.
