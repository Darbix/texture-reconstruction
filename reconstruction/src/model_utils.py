# model_utils.py

import torch
import torch.nn as nn
from torchvision.models import vgg19


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

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
