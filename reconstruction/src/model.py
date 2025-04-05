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
    """UNet architecture model with ResNet34 backbone"""
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
        h_pad = (32 - (H % 32)) % 32
        w_pad = (32 - (W % 32)) % 32
        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')
        
        x = self.unet(x)
        
        # Remove the padding to restore the original size
        x = x[:, :, :H, :W]
        
        return x




class MVTRN_UNet_Separated(nn.Module):
    """UNet-based model with per-view convolutional processing before fusion"""
    def __init__(self, num_views, backbone='resnet34', pretrained=True):
        super(MVTRN_UNet_Separated, self).__init__()
        self.num_views = num_views
        
        # Feature extraction per view
        self.view_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # Keep spatial size
                nn.ReLU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Output 64 channels per view
                nn.ReLU()
            ) for _ in range(num_views)
        ])
        
        # UNet model (updated input channels to match feature extraction output)
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=32 * num_views, # Concatenated input channels
            classes=3 # Output image channels
        )

    def forward(self, x):
        # Process a batch of patches (sets of view crops at the same position)
        # B: batch size, N: number of views, C: channels, H: height, W: width
        B, N, C, H, W = x.shape

        # Process each view separately through its own feature extractor
        processed_views = []
        for i in range(N):
            # Extract i-th view of shape: (B, 3, H, W)
            view = x[:, i]
            processed_view = self.view_extractors[i](view) # (B, 64, H, W)
            
            # Ensure height and width are multiples of 32 for UNet compatibility
            # Expects no size change in view_extractors
            h_pad = (32 - (H % 32)) % 32
            w_pad = (32 - (W % 32)) % 32
            processed_view = F.pad(processed_view, (0, w_pad, 0, h_pad),
                mode='reflect')
 
            processed_views.append(processed_view)

        # Concatenate processed views along the channel dimension
        x = torch.cat(processed_views, dim=1) # (B, 64 * N, H, W)

        x = self.unet(x)

        # Remove padding to restore original size
        x = x[:, :, :H, :W]

        return x




class MVTRN_UNet_MiT(nn.Module):
    """UNet architecture model with Mix Vision Transformer (MiT) backbone"""
    def __init__(self, num_views, backbone='MiT_b2', pretrained=True):
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
        view_enc_channels = 16

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

        # Upsampling 4x to match UNet resolution
        self.upsample = nn.ConvTranspose2d(
            view_enc_channels * num_views,
            view_enc_channels * num_views,
            kernel_size=8, stride=4, padding=2
        )

        # Reshape processed views (residuum) to match the original input
        self.match_channels = nn.Conv2d(
            view_enc_channels * num_views, 3 * num_views, kernel_size=1
        )

        # UNet backbone
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
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
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 1st 2x downsampling

            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 2nd 2x downsampling

            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)
