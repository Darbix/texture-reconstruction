# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import segmentation_models_pytorch as smp


class MVTRN(nn.Module):
    """Multi-View Texture Reconstruction Network"""
    def __init__(self, num_views, upscale_factor=1):
        super(MVTRN, self).__init__()
        self.num_views = num_views
        self.upscale_factor = upscale_factor

        # Shared feature extractor for all views
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # RGB
            nn.ReLU(inplace=True),
            *[ResidualBlock(64) for _ in range(3)]
        )

        # Fusion layer
        self.fusion = nn.Conv2d(num_views * 64, 64, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(5)]
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64 * (upscale_factor ** 2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1) # RGB
        )

    def forward(self, x):
        B, N, C, H, W = x.shape

        # Apply the feature extraction to each view independently (in parallel)
        x = x.view(B * N, C, H, W)  # Merge batch and view dimensions
        x = self.feature_extractor(x)
        x = x.view(B, N * 64, H, W) # Restore a batch dimension and stack view features

        x = self.fusion(x)

        # Super-resolution part
        residual = x
        x = self.res_blocks(x)
        x += residual
        x = self.upsample(x)

        return x


class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)



class MVTRN_UNet(nn.Module):
    """UNet architecture model for image enhancing"""
    def __init__(self, num_views, backbone='resnet34', pretrained=True):
        super(MVTRN_UNet, self).__init__()
        self.num_views = num_views
        
        # segmentation_models_python UNet implementation
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=3 * num_views, # Input 3 channels per 1 view
            classes=3                  # Output 3 channel image
        )

    def forward(self, x):
        # Process a batch of patches (sets of view crops at the same position)
        # B: batch size, N: number of views, C: channels, H: height, W: width
        B, N, C, H, W = x.shape

        # Merge view dimension with channels to represent input
        x = x.view(B, C * N, H, W)

        # Ensure height and width are multiples of 32 by padding
        h_pad = (-H) % 32
        w_pad = (-W) % 32
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        
        x = self.unet(x)
        
        # Remove the padding to restore the original size
        x = x[:, :, :H, :W]
        
        return x
