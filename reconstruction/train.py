# train.py

import os
import gc
import sys
import time
import random
import argparse
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader

from utils import attach_tiles_to_images, plot_epoch_images, resize_max_side, \
    resize_max_side, load_data
from model import MVTRN, MSELoss, save_checkpoint, load_checkpoint
from dataset import MultiViewDataset
from test_plot import test_plot


EPOCH_LOG_INTERVAL = 1 # Epoch logging interval
SAMPLE_N = 0           # Index in the batch sample to visualize in a plot 



def parse_args():
    parser = argparse.ArgumentParser(description="Multi-View Texture Reconstruction Network training")
    # Paths
    parser.add_argument('--data_path', type=str, required=True, help="Path to a dataset")
    parser.add_argument('--output_dir', type=str,default=f'output_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}', help="Output directory")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to a checkpoint .pth file")
    # Training parameters
    parser.add_argument('--num_views', type=int, default=30, help="Number of low-resolution views")
    parser.add_argument('--num_scenes', type=int, default=-1, help="Number of scenes to use for training")
    parser.add_argument('--input_resolution', type=int, default=512, help="Input resolution")
    parser.add_argument('--tile_size', type=int, default=256, help="Tile size")
    parser.add_argument('--tile_stride', type=int, help="Tile size")
    parser.add_argument('--upscale_factor', type=int, default=1, help="Super-resolution scale factor")
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.000075, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--num_workers', type=int, default=2, help="Number of workers")
    parser.add_argument('--max_workers_loading', type=int, default=1, help="Number of workers for loading view images")
    return parser.parse_args()


def train_mvtrn(model, dataloader, criterion, optimizer, device, num_epochs=10,
                start_epoch=0, loss_hist=[], output_dir="", tile_size=256,
                num_scenes=None):
    """Train loop for the MVTRN model training"""
    
    model = model.to(device)
    criterion = criterion.to(device)
    model.train()

    start_time = time.time()
    max_n = num_scenes - 1

    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0
        # Batch index of the random epoch sample to plot
        random_sample_idx = random.randint(0, max_n)
        # Images composed from tiles used for plotting
        rnd_lr_img, rnd_sr_img, rnd_hr_img = None, None, None

        # Load batches as: [B, N, C, TH, TW], [B, C, SH, SW], [B, S, TY, TX, T, PH, PW]
        # B: batches, N: num of views, C: channels, T<H,W>: tile height and width,
        # S<H,W>: GT texture height and width (resized to the model output size)
        # S: scene index, T<Y,X>: tile pixel position in the input image, T: num of scene tiles,
        # PH: original input height with padding, PW: original input width with padding
        for batch_idx, (b_lr_imgs, b_hr_texture_tile, b_s_tile) in enumerate(dataloader):
            optimizer.zero_grad()
            tile_loss = 0

            # Get an index of a batch of tiles in the input image (pixel location)
            tile_y, tile_x = b_s_tile[0, 1:3]
            tile_y, tile_x = (tile_y.item(), tile_x.item())

            # Convert input and output using device
            b_lr_imgs = b_lr_imgs.to(device, non_blocking=True)
            b_hr_texture_tile = b_hr_texture_tile.to(device, non_blocking=True)

            # Process the input batch of scene sets (views as tiles)
            b_sr_img_tile = model(b_lr_imgs)


            # Attach a tile from LR, SR and HR images to composed images
            # used to plot a random sample scene each epoch
            scene_idx = b_s_tile[0][0]
            if(scene_idx == random_sample_idx):
                # LR image
                img_size = (b_lr_imgs.shape[2], b_s_tile[SAMPLE_N][4], b_s_tile[SAMPLE_N][5])
                rnd_lr_img = attach_tiles_to_images(
                    rnd_lr_img, img_size=img_size, tile=b_lr_imgs[SAMPLE_N][0],
                    tile_x=tile_x, tile_y=tile_y
                )
                # SR image
                sr_factor = b_sr_img_tile.shape[3] / b_lr_imgs.shape[4] # SR_TW / LR_TW
                img_size = (b_sr_img_tile.shape[1], *img_size[1:3])
                rnd_sr_img = attach_tiles_to_images(
                    rnd_sr_img, img_size=img_size, tile=b_sr_img_tile[SAMPLE_N],
                    tile_x=tile_x, tile_y=tile_y, sr_factor=sr_factor
                )
                # HR image (GT texture)
                rnd_hr_img = attach_tiles_to_images(
                    rnd_hr_img, img_size=img_size, tile=b_hr_texture_tile[SAMPLE_N],
                    tile_x=tile_x, tile_y=tile_y, sr_factor=sr_factor
                )

            # Compute loss for this tile
            tile_loss = criterion(b_sr_img_tile, b_hr_texture_tile)

            # Normalize loss by a number of tiles corresponding to the scene
            tile_loss = tile_loss / b_s_tile[0][3] # TODO sum over a whole batch
            tile_loss.backward()
            epoch_loss += tile_loss.item()
            optimizer.step()

            # Free memory
            del b_lr_imgs, b_hr_texture_tile, b_sr_img_tile, tile_loss
            torch.cuda.empty_cache()
            gc.collect()

        # Time run yet in minutes
        curr_run_time = (time.time() - start_time) / 60
        print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}],",
              f"Loss: {epoch_loss / len(dataloader):.6f},",
              f"Time taken: {curr_run_time:.1f} minutes")

        loss_hist.append(epoch_loss / len(dataloader))

        # Save checkpoint every EPOCH_LOG_INTERVAL epochs and plot a random sample
        if(epoch + 1) % EPOCH_LOG_INTERVAL == 0:
            checkpoint_path = os.path.join(output_dir, f"mvtrn_epoch{epoch + 1}.pth")
            save_checkpoint(model, checkpoint_path, optimizer, epoch + 1, loss_hist)
            print(f"Model checkpoint saved to {checkpoint_path}")

            sample_plot_path = os.path.join(output_dir, f'plot_epoch_{epoch + 1}.png')
            plot_epoch_images(rnd_lr_img, rnd_sr_img, rnd_hr_img, sample_plot_path)
            print(f"Sample plot saved to {sample_plot_path}")

    return loss_hist



if __name__ == "__main__":
    print("hi")
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transform_view = T.Compose([
        T.Lambda(lambda img: resize_max_side(img, args.input_resolution))
    ])
    transform_texture = T.Compose([
        T.Lambda(lambda img: resize_max_side(img, args.input_resolution * args.upscale_factor))
    ])
    transform_tile = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5, 0.5]),  # Normalize to [-1, 1]
    ])


    print("Loading data from:", args.data_path)
    # Load the dataset using the new MultiViewDataset
    dataset = MultiViewDataset(data_path=args.data_path,
                               transform_view=transform_view,
                               transform_texture=transform_texture,
                               transform_tile=transform_tile,
                               n=args.num_views, tile_size=args.tile_size,
                               s=args.num_scenes, tile_stride=args.tile_stride,
                               input_max_res=args.input_resolution,
                               num_workers=args.num_workers,
                               max_workers_loading=args.max_workers_loading)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, persistent_workers=True)


    # ----- Model setup -----
    model = MVTRN(num_views=args.num_views, upscale_factor=args.upscale_factor)
    # criterion = PerceptualLoss()  # Perceptual loss for better visual quality
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    start_epoch = 0
    loss_hist = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Possible training resume from a checkpoint
    if args.checkpoint_path:
        checkpoint = load_checkpoint(model, args.checkpoint_path, optimizer, device)
        model, optimizer, start_epoch, loss_hist = checkpoint

        print(f"Resumed training from epoch {start_epoch}")



    # ----- Model training -----
    loss_hist = train_mvtrn(model, dataloader, criterion, optimizer,
                            device, args.num_epochs, start_epoch, loss_hist,
                            args.output_dir, tile_size=args.tile_size,
                            num_scenes=(args.num_scenes if args.num_scenes > 0 else dataset.num_of_scenes()))

    plt.plot(loss_hist)
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))



    # ----- Test inference with texture comparison -----
    # TODO
    test_scene_path = os.path.join(args.data_path, "01d53fec-15e9-4dbd-8989-f11051caff25")

    # Load a target texture
    target_texture_file = [
        f for f in os.listdir(test_scene_path) if f.endswith(('.png', '.jpg', '.jpeg'))
    ][0]
    # Load a list of tiles as data
    list_tiles = load_data(
        os.path.join(test_scene_path, "color_imgs"),
        os.path.join(test_scene_path, target_texture_file),
        transform_view, transform_texture, transform_tile, args.num_views,
        tile_size=args.tile_size, input_max_res=args.input_resolution
    )

    # Load a model
    cnn_model = MVTRN(num_views=args.num_views, upscale_factor=args.upscale_factor)
    optimizer = optim.Adam(cnn_model.parameters(), lr=args.learning_rate)
    cnn_model, _, _, _ = load_checkpoint(
        cnn_model,
        os.path.join(args.output_dir, f"mvtrn_epoch{start_epoch + args.num_epochs}.pth"),
        optimizer, device
    )

    cnn_model.to(device)
    cnn_model.eval()

    test_plot(cnn_model, list_tiles, device, args.tile_size,
              "plot.png", path_individuals=args.output_dir)
    