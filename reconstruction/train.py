# train.py

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'config')))

import gc
import time
import random
import argparse
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import plot_patches, load_img_patches, plot_random_val_patch, \
    compose_sr_lr_hr, get_patch_transform
from model_utils import MSELoss, save_checkpoint, load_checkpoint, setup_model
from dataset import MultiViewDataset
import config


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-View Texture Reconstruction Network training")
    # Paths
    parser.add_argument('--data_path', type=str, required=True, help="Path to a dataset")
    parser.add_argument('--output_dir', type=str,default=f"output_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}", help="Output directory")
    parser.add_argument('--checkpoint_path', type=str, default=None, help="Path to a checkpoint .pth file")
    # Training parameters
    parser.add_argument('--num_views', type=int, default=6, help="Number of views in a patch set")
    parser.add_argument('--input_resolution', type=int, default=-1, help="Input resolution")
    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=0.000075, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--num_workers', type=int, default=1, help="Number of workers")
    # TODO remove?
    parser.add_argument('--max_workers_loading', type=int, default=1, help="Number of workers for loading view images")
    # Other
    parser.add_argument('--model_type', type=str, required=False, help="'UNET' or 'DEFAULT' MVTRN")
    return parser.parse_args()


def evaluate(model, dataloader, criterion, device, output_dir, epoch):
    model.eval()
    total_loss = 0

    rnd_lr_img, rnd_sr_img, rnd_hr_img = None, None, None

    with torch.no_grad():
        for batch_idx, (b_lr_imgs, b_hr_texture_tile, b_info) in enumerate(dataloader):
            b_lr_imgs = b_lr_imgs.to(device, non_blocking=True)
            b_hr_texture_tile = b_hr_texture_tile.to(device, non_blocking=True)

            b_sr_img_tile = model(b_lr_imgs)
            tile_loss = criterion(b_sr_img_tile, b_hr_texture_tile)

            total_loss += tile_loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_mvtrn(
    model, train_dataloader, val_dataloader, criterion, optimizer,
    device, num_epochs=10, start_epoch=0, loss_hist=[], output_dir=""):
    
    """Train loop for the MVTRN model training"""
    
    model = model.to(device)
    criterion = criterion.to(device)

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        epoch_loss = 0

        # B: batch, N: number of patches, C: channels, H & W: patch size,
        # I: scene index, PY & PX: patch pixel position, TH & TW: whole image size, PS: patch size 
        # [B, N, C, H, W], [B, C, H, W], [B, I, PY, PX, TH, TW, PS]
        for batch_idx, (b_lr_patches, b_hr_texture_patch, b_patch_info) in enumerate(train_dataloader):
            optimizer.zero_grad()
            patch_loss = 0

            # Each 5th batch plot info
            if((batch_idx + 1) % 5 == 0 or batch_idx == 0):
                curr_run_time = (time.time() - start_time) / 60
                print(f"{batch_idx + 1}/{len(train_dataloader)},",
                      f"{curr_run_time:.1f} minutes")

            # Convert input and output using device
            b_lr_patches = b_lr_patches.to(device, non_blocking=True)
            b_hr_texture_patch = b_hr_texture_patch.to(device, non_blocking=True)

            # Process the input batch of scene sets (views as tiles)
            b_sr_patch = model(b_lr_patches)
            
            # Compute loss for this tile
            patch_loss = criterion(b_sr_patch, b_hr_texture_patch)

            patch_loss.backward()
            epoch_loss += patch_loss.item()
            optimizer.step()

            # Free memory
            del b_lr_patches, b_hr_texture_patch, b_sr_patch, patch_loss
            torch.cuda.empty_cache()
            gc.collect()

        loss_hist.append(epoch_loss / len(train_dataloader))

        # Print a time run yet in minutes after an epoch
        curr_run_time = (time.time() - start_time) / 60
        print(f"Epoch [{epoch+1}/{start_epoch + num_epochs}],",
              f"Loss: {epoch_loss / len(train_dataloader):.6f},",
              f"Time taken: {curr_run_time:.1f} minutes")

        # Save a checkpoint every EPOCH_LOG_INTERVAL epochs
        if(epoch + 1) % config.EPOCH_LOG_INTERVAL == 0:
            checkpoint_path = os.path.join(output_dir, f"mvtrn_epoch_{epoch + 1}.pth")
            save_checkpoint(model, checkpoint_path, optimizer, epoch + 1, loss_hist)
            print(f"Model checkpoint saved to {checkpoint_path}")

        # Plot a random validation dataset patch
        model.eval()
        with torch.no_grad():
            val_path = os.path.join(output_dir, f'plot_val_patch_epoch_{epoch + 1}.jpg')
            plot_random_val_patch(model, val_dataloader, val_path, device)
            print(f"Random validation patch plot saved to {val_path}")

        # Evaluate the model on the validation dataset
        if (epoch + 1) % config.EPOCH_EVAL_INTERVAL == 0:
            print(f"Evaluation on validation dataset at epoch {epoch + 1}")
            val_loss = evaluate(model, val_dataloader, criterion, device, output_dir, epoch)
            print(f"Validation loss: {val_loss:.6f}")
    
    return loss_hist



if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Load train and val datasets -----
    print("Loading data from:", args.data_path)
    train_dataset = MultiViewDataset(
        data_path=args.data_path, transform_patch=get_patch_transform(),
        n_patches=args.num_views,
        input_max_res=args.input_resolution,
        split_ratio=0.8, train=True)

    val_dataset = MultiViewDataset(
        data_path=args.data_path, transform_patch=get_patch_transform(),
        n_patches=args.num_views,
        input_max_res=args.input_resolution,
        split_ratio=0.8, train=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True)

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True)


    # ----- Model setup -----
    model = setup_model(args.model_type, num_views=args.num_views)

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
    loss_hist = train_mvtrn(
        model, train_dataloader, val_dataloader, criterion, optimizer,
        device, args.num_epochs, start_epoch, loss_hist, args.output_dir
    )

    plt.plot(loss_hist)
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))


    # ----- Test inference -----
    # Load weights of a trained model
    cnn_model = setup_model(args.model_type, num_views=args.num_views)
    
    optimizer = optim.Adam(cnn_model.parameters(), lr=args.learning_rate)
    device = torch.device("cpu")
    cnn_model, _, _, _ = load_checkpoint(
        cnn_model,
        os.path.join(args.output_dir, f"mvtrn_epoch_{start_epoch + args.num_epochs}.pth"),
        optimizer, device
    )
    cnn_model.to(device)
    cnn_model.eval()

    # Enhance the test image loaded as patches from the test scene 
    test_scene_path = os.path.join(args.data_path, "01d53fec-15e9-4dbd-8989-f11051caff25")
    output_image, view_image, texture_image = compose_sr_lr_hr(
        cnn_model, test_scene_path, args.num_views, max_size=5000)

    output_names = ["plot_output_image.jpg", "plot_view_image.jpg", "plot_texture_image.jpg"]
    plt.imsave(os.path.join(args.output_dir, output_names[0]), output_image)
    plt.imsave(os.path.join(args.output_dir, output_names[1]), view_image)
    plt.imsave(os.path.join(args.output_dir, output_names[2]), texture_image)
    print(f"Output images saved to {', '.join(output_names)}")
