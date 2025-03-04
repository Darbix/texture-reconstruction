# Baseline multi-view texture reconstruction network for image resolution and quality enhancement

Example run:

```
python train.py --data_path "path/to/aligned/data" --num_views 30 --num_scenes 150 --input_resolution 1024 --tile_size 256 --learning_rate 0.000075 --num_epochs 10 --num_workers 4 --max_workers_loading 2 --output_dir "output/path" --checkpoint_path ""
```