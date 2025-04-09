# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import segmentation_models_pytorch as smp


class MVTRN_UNet(nn.Module):
    """UNet architecture model with ResNet34 backbone"""
    def __init__(self, num_views, backbone='resnet34', pretrained=True):
        super(MVTRN_UNet, self).__init__()
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




class MVTRN_UNetPlusPlus(nn.Module):
    """UNet++ architecture model with ResNet50 backbone"""
    def __init__(self, num_views, backbone='resnet50', pretrained=True):
        super(MVTRN_UNetPlusPlus, self).__init__()
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




class MVTRN_UNet_MiT(nn.Module):
    """UNet architecture model with Mix Vision Transformer (MiT) backbone"""
    def __init__(self, num_views, backbone='mit_b2', pretrained=True):
        super(MVTRN_UNet_MiT, self).__init__()
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




class MVTRN_UNet_Attention(nn.Module):
    """UNet with ResNet34 backbone and attention-based multi-view fusion"""
    def __init__(self, num_views, backbone='resnet34', pretrained=True):
        super().__init__()
        self.num_views = num_views
        # Number of output channels per processed view of individual encoders
        view_enc_channels = 8

        # Encoders for each view
        self.view_encoders = nn.ModuleList(
            [ViewEncoder(3, view_enc_channels) for _ in range(num_views)]
        )

        # Channel-wise fusion weighting
        self.view_weighting = ChannelWiseViewWeighting(num_views, view_enc_channels)
        # Spatial attention for input improvement
        self.attn = nn.MultiheadAttention(
            embed_dim=view_enc_channels * num_views, num_heads=1
        )

        # Gradually upsamples features 8x to match x input resolution to U-Net
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                view_enc_channels * num_views, view_enc_channels * num_views,
                kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                view_enc_channels * num_views, view_enc_channels * num_views,
                kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                view_enc_channels * num_views, view_enc_channels * num_views,
                kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        # Reshape processed views (residuum) to match the original input
        self.match_channels = nn.Conv2d(
            view_enc_channels * num_views, 3 * num_views, kernel_size=1
        )

        # UNet backbone
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            activation='tanh', # Converts output to the range [-1, 1]
            in_channels=3 * num_views,
            classes=3
        )

    def forward(self, x):
        # x: [B, N, C, H, W]
        B, N, C, H, W = x.shape

        # Encode each view separately
        processed_views = [] # List of N views processed by individual encoders
        for i in range(N):
            view = x[:, i, :, :, :] # Extract i-th view [B, C, H, W]
            view_encoded = self.view_encoders[i](view)
            processed_views.append(view_encoded) # [B, C_enc, H_enc, W_enc]

        # Fuse using channel-wise attention
        weighted_views = self.view_weighting(processed_views) # [B, C_enc*N, H_enc, W_enc]

        # Apply attention over flattened spatial dimensions
        B, C_enc_N, H_enc, W_enc = weighted_views.shape
        weighted_views = weighted_views.view(B, C_enc_N, -1).permute(2, 0, 1) # [H_enc*W_enc, B, C_enc_N]
        
        fused_views, _ = self.attn(weighted_views, weighted_views, weighted_views)
        fused_views = fused_views.permute(1, 2, 0).view(B, C_enc_N, H_enc, W_enc) # [B, C_enc_N, H_enc, W_enc]

        # Upsample and match [B, N*C, H, W]
        fused_views = self.upsample(fused_views)
        fused_residual = self.match_channels(fused_views)
        x_flat = x.view(B, -1, H, W) # [B, N*C, H, W]

        # Residual addition and final UNet processing
        x_improved = x_flat + fused_residual
        x_improved = self.unet(x_improved)

        return x_improved

class ChannelWiseViewWeighting(nn.Module):
    """Channel-wise weighting fusion"""
    def __init__(self, num_views, channels):
        super().__init__()
        self.num_views = num_views
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channels * num_views, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels * num_views, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, features):
        # features: list of N tensors [B, C, H, W]
        B, C, H, W = features[0].shape
        x_cat = torch.cat(features, dim=1) # [B, N*C, H, W]
        view_weights = self.attn_conv(x_cat).view(B, self.num_views, C, H, W) # [B, N, C, H, W]

        weighted_views = torch.stack(
            # Weight each view by n-th weight
            [view_f * view_weights[:, n] for n, view_f in enumerate(features)],
            dim=1) # [B, N, C, H, W]
        return weighted_views.view(B, -1, H, W) # [B, N*C, H, W]

class ViewEncoder(nn.Module):
    """Individual encoder for each view image with inline pooling"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 1st 2x downsampling

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 2nd 2x downsampling

            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 3rd 2x downsampling

            nn.Conv2d(8, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)




class MVTRN_EDSR(nn.Module):
    """
    Multi-view super-resolution model inspired by single-view EDSR architecture
    for super-resolution (https://github.com/LimBee/NTIRE2017).
    """
    def __init__(self, num_views, num_residual_blocks=32, num_channels=64):
        super(MVTRN_EDSR, self).__init__()

        self.num_views = num_views

        # Initial convolution to process multi-view input
        self.initial_conv = nn.Conv2d(3 * num_views, num_channels, kernel_size=3, padding=1)

        # Stack of residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(num_channels) for _ in range(num_residual_blocks)]
        )

        # Final convolution layer
        self.final_conv = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)  # Output 3 channels (RGB)

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




class MVTRN_EfficientNet_MANet(nn.Module):
    """MAnet with EfficientNet backbone for multi-view reconstruction model"""
    def __init__(self, num_views, backbone='efficientnet-b5', pretrained=True):
        super(MVTRN_EfficientNet_MANet, self).__init__()
        self.num_views = num_views
        
        self.manet = smp.MAnet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            activation='tanh', # Converts output to the range [-1, 1]
            in_channels=3 * num_views,
            classes=3
        )

    def forward(self, x):
        # x: [B, N, C, H, W]
        B, N, C, H, W = x.shape

        # Merge the views with channels
        x = x.view(B, C * N, H, W)

        x = self.manet(x)
        
        return x




class MVTRN_SegFormer(nn.Module):
    """MiT & Segformer model for multi-view reconstruction"""
    def __init__(self, num_views, backbone='mit_b2', pretrained=True):
        super(MVTRN_SegFormer, self).__init__()
        self.num_views = num_views
        
        self.segformer = smp.Segformer(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            activation='tanh', # Converts output to the range [-1, 1]
            in_channels=3 * num_views,
            classes=3
        )

    def forward(self, x):
        # x: [B, N, C, H, W]
        B, N, C, H, W = x.shape

        # Merge the views with channels
        x = x.view(B, C * N, H, W)

        y = self.segformer(x)

        return y




class MVTRN_UNet_Swin(nn.Module):
    """Swin-V2 & U-Net model"""
    def __init__(self, num_views, backbone='tu-swinv2_base_window16_256', pretrained=True):
        super(MVTRN_UNet_Swin, self).__init__()
        self.num_views = num_views
        
        self.swin_unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            activation='tanh', # Converts output to the range [-1, 1]
            in_channels=3 * num_views,
            classes=3
        )

    def forward(self, x):
        # x: [B, N, C, H, W]
        B, N, C, H, W = x.shape
        
        x = x.view(B, C * N, H, W)

        # Resize input to (256, 256) for SwinV2 backbone
        x_resized = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        y = self.swin_unet(x_resized)

        # Upsample the output back to original height and width
        y_upsampled = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)

        return y_upsampled
