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




class SEBlock(nn.Module):
    """Squeeze-and-Excitation (SE) block for channel-wise attention"""
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        # Adaptive pooling with output size (1, 1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # Channel size reduction
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            # Output attention weights between 0 and 1
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, _, _ = x.shape
        # Global average pooling across height and width
        se_weight = self.global_avg_pool(x).view(B, C)
        # Attention vector reshaped to (B, C, 1, 1)
        se_weight = self.fc(se_weight).view(B, C, 1, 1)
        return x * se_weight


class SelfAttentionBlock(nn.Module):
    """Memory-efficient self-attention block with downsampling and upsampling"""
    def __init__(self, in_channels, num_heads=2, scale_factor=2):
        super(SelfAttentionBlock, self).__init__()

        self.num_heads = num_heads
        self.in_channels = in_channels
        self.scale_factor = scale_factor

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)

        # Feed-forward layers
        self.fc1 = nn.Linear(in_channels, in_channels * 2)
        self.fc2 = nn.Linear(in_channels * 2, in_channels)

        # Upsampling to restore original size
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

    def forward(self, x):
        B, C, H, W = x.shape

        # Downsample using interpolation
        H_reduced, W_reduced = H // self.scale_factor, W // self.scale_factor
        x_down = F.interpolate(x, size=(H_reduced, W_reduced), mode='bilinear', align_corners=False)

        # Reshape for self-attention
        x_attn = x_down.view(B, C, -1).permute(0, 2, 1) # Shape: (B, H'W', C)

        # Apply multi-head attention without mixed precision (no autocast)
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)

        # Residual connection
        attn_output = attn_output + x_attn
        attn_output = self.fc2(F.relu(self.fc1(attn_output))) + attn_output # Ensure shape consistency

        # Reshape back and upsample to original size
        attn_output = attn_output.permute(0, 2, 1).view(B, C, H_reduced, W_reduced)
        output = self.upsample(attn_output)

        return output


class MVTRN_UNet_Attention(nn.Module):
    """UNet architecture with self-attention and spatial attention"""
    def __init__(self, num_views, backbone='resnet34', pretrained=True):
        super(MVTRN_UNet_Attention, self).__init__()
        self.num_views = num_views
        in_channels = 3 * num_views  # Each view has 3 channels
        
        # Add channel-wise attention (SE Block)
        self.se_block = SEBlock(in_channels)

        # Add self-attention block to enhance spatial attention
        self.self_attention = SelfAttentionBlock(in_channels)
        
        # UNet with ResNet backbone
        self.unet = smp.Unet(
            encoder_name=backbone,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=3 # Output 3-channel image
        )

    def forward(self, x):
        B, N, C, H, W = x.shape

        # Merge view dimension with channels to represent input
        x = x.view(B, C * N, H, W)

        # Apply SE block (channel-wise attention)
        x = self.se_block(x)

        # Apply self-attention to enhance spatial relationships
        x = self.self_attention(x)

        # Ensure height and width are multiples of 32 by padding
        h_pad = (32 - (H % 32)) % 32
        w_pad = (32 - (W % 32)) % 32

        x = F.pad(x, (0, w_pad, 0, h_pad), mode='reflect')

        x = self.unet(x)

        # Remove the padding to restore original size
        x = x[:, :, :H, :W]

        return x
