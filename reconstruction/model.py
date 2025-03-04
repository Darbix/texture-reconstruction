# model.py

import torch
import torch.nn as nn
from torchvision.models import vgg19



class MVTRN(nn.Module):
    """Multi-View Texture Reconstruction Network"""
    def __init__(self, num_views, upscale_factor):
        super(MVTRN, self).__init__()
        self.num_views = num_views
        self.upscale_factor = upscale_factor

        # Shared feature extractor for all views
        self.feature_extractor = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), # RGB
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1), # RGBA
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
            # nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1) # RGB
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1) # RGBA
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

        # Model eval mode
        if not self.training:
            # Round alpha values for evaluation inference mode
            # Get the alpha channel (4th channel)
            alpha = x[:, 3, :, :]

            # Normalize to [0, 1] using sigmoid
            alpha = torch.sigmoid(alpha)

            # Round the alpha channel and map it back to [-1, 1]
            thresholded_alpha = torch.round(alpha) * 2 - 1

            # Replace the alpha channel with the rounded values
            x[:, 3, :, :] = thresholded_alpha

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


class PerceptualLoss(nn.Module):
    """Perceptual loss function"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)
        return self.criterion(output_features, target_features)


class MSELoss(nn.Module):
    """MSE loss function"""
    def __init__(self):
        super(MSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        return self.criterion(output, target)


def load_checkpoint(model, checkpoint_path, optimizer, device):
    """Loads a checkpoint of a model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    loss_hist = checkpoint['loss_hist']

    if(optimizer):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer internal states to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return model, optimizer, epoch, loss_hist


def save_checkpoint(model, checkpoint_path, optimizer, epoch, loss_hist):
    """Saves the model, optimizer state dict, epoch and loss history"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_hist': loss_hist,
    }, checkpoint_path)
