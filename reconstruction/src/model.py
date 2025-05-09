# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import segmentation_models_pytorch as smp


class MVTRN_ResNet34_UNet(nn.Module):
    """ResNet34_U-Net model"""
    def __init__(self, num_views, backbone='resnet34', pretrained=True):
        super(MVTRN_ResNet34_UNet, self).__init__()
        self.num_views = num_views
        
        # segmentation_models_python UNet implementation
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            activation='tanh', # Converts output to the range [-1, 1]
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
        h_pad = (32 - (H % 32)) % 32
        w_pad = (32 - (W % 32)) % 32
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        
        x = self.unet(x)
        
        # Remove the padding to restore the original size
        x = x[:, :, :H, :W]
        
        return x



class MVTRN_MiTB4_UNet(nn.Module):
    """MiTB4_U-Net model with Mix Vision Transformer (MiT) backbone"""
    def __init__(self, num_views, backbone='mit_b4', pretrained=True):
        super(MVTRN_MiTB4_UNet, self).__init__()
        self.num_views = num_views
        
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
        h_pad = (32 - (H % 32)) % 32
        w_pad = (32 - (W % 32)) % 32
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        
        x = self.unet(x)
        
        # Remove the padding to restore the original size
        x = x[:, :, :H, :W]
        
        return x



class MVTRN_WF_ResNet34_UNet(nn.Module):
    """WF_ResNet34_U-Net model with Weighted Fusion of views"""
    def __init__(self, num_views, backbone='resnet34', pretrained=True):
        super(MVTRN_WF_ResNet34_UNet, self).__init__()
        self.num_views = num_views
        self.encoder_channels = 8  # Matches your ViewEncoder output

        # Use your original downsampling + pooling encoder
        self.view_encoders = nn.ModuleList([
            ViewEncoder(3, self.encoder_channels) for _ in range(num_views)
        ])

        # Squeeze-Excite style fusion: 1 weight per view
        self.fusion_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, C, 1, 1]
            nn.Flatten(start_dim=1),  # [B, C]
            nn.Linear(self.encoder_channels, self.encoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_channels // 2, 1),
            nn.Sigmoid()
        )

        # Final U-Net with stacked weighted views
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            activation='tanh',
            in_channels=3 * num_views,
            classes=3
        )

    def forward(self, x):
        # x: [B, N, 3, H, W]
        B, N, C, H, W = x.shape
        weighted_views = []

        for i in range(N):
            view = x[:, i]  # [B, 3, H, W]
            enc_feat = self.view_encoders[i](view)  # [B, C_enc, H/8, W/8]
            weight = self.fusion_weights(enc_feat)  # [B, 1]
            weighted_view = view * weight.view(B, 1, 1, 1)
            weighted_views.append(weighted_view)

        # Stack all weighted views along channel dimension
        x_weighted = torch.cat(weighted_views, dim=1)  # [B, 3*N, H, W]

        # Pad to match UNet requirement
        h_pad = (32 - H % 32) % 32
        w_pad = (32 - W % 32) % 32
        x_weighted = F.pad(x_weighted, (0, w_pad, 0, h_pad), mode='reflect')

        out = self.unet(x_weighted)
        return out[:, :, :H, :W]

class ViewEncoder(nn.Module):
    """Separated view pre-encoder"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 2x downsize
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4x downsize
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x downsize
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)



class MVTRN_ResNet101_UNetPlusPlus(nn.Module):
    """ResNet101_UNet++ model"""
    def __init__(self, num_views, backbone='resnet101', pretrained=True):
        super(MVTRN_ResNet101_UNetPlusPlus, self).__init__()
        self.num_views = num_views
        
        self.unet = smp.UnetPlusPlus(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            activation='tanh', # Converts output to the range [-1, 1]
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
        h_pad = (32 - (H % 32)) % 32
        w_pad = (32 - (W % 32)) % 32
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        
        x = self.unet(x)
        
        # Remove the padding to restore the original size
        x = x[:, :, :H, :W]
        
        return x



class MVTRN_MV_EDSR(nn.Module):
    """
    MV_EDSR model
    Multi-view super-resolution model inspired by single-view EDSR architecture
    for super-resolution (https://github.com/LimBee/NTIRE2017).
    """
    def __init__(self, num_views, num_residual_blocks=32, num_channels=64):
        super(MVTRN_MV_EDSR, self).__init__()

        self.num_views = num_views

        # Initial convolution to process multi-view input
        self.initial_conv = nn.Conv2d(3 * num_views, num_channels, kernel_size=3, padding=1)

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        )

        # Final convolution layer
        self.final_conv = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, N, C, H, W]
        B, N, C, H, W = x.shape

        # Merge the views with channels
        x = x.view(B, C * N, H, W)

        x = self.initial_conv(x)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Final convolution for RGB image reconstruction
        x = self.final_conv(x)

        x = torch.tanh(x) # Converts output to the range [-1, 1]

        return x

class ResidualBlock(nn.Module):
    """Residual block"""
    def __init__(self, channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual
