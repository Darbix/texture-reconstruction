# model_utils.py

import torch
import torch.nn as nn
from torchvision.models import vgg19

from model import MVTRN_UNet, MVTRN_EDSR, MVTRN_EfficientNet_MANet, \
    MVTRN_UNet_MiT, MVTRN_UNet_Attention, MVTRN_UNetPlusPlus, MVTRN_Segformer
import config


def setup_model(model_type, num_views=6):
    model = None
    if(model_type == config.ModelType.UNET.value):
        print("MVTRN_UNet")
        model = MVTRN_UNet(num_views=num_views)
    elif(model_type == config.ModelType.UNETPLUSPLUS.value):
        print("MVTRN_UNetPlusPlus")
        model = MVTRN_UNetPlusPlus(num_views=num_views)
    elif(model_type == config.ModelType.UNET_ATTENTION.value):
        print("MVTRN_UNet_Attention")
        model = MVTRN_UNet_Attention(num_views=num_views)
    elif(model_type == config.ModelType.UNET_MIT.value):
        print("MVTRN_UNet_MiT")
        model = MVTRN_UNet_MiT(num_views=num_views)
    elif(model_type == config.ModelType.MVEDSR.value):
        print("MVTRN_EDSR")
        model = MVTRN_EDSR(num_views=num_views)
    elif(model_type == config.ModelType.EFFIC_MANET.value):
        print("MVTRN_EfficientNet_MANet")
        model = MVTRN_EfficientNet_MANet(num_views=num_views)
    elif(model_type == config.ModelType.SEGFORMER.value):
        print("MVTRN_Segformer")
        model = MVTRN_Segformer(num_views=num_views)
    else:
        print(f"Model {model_type} does not exist")
    return model


def model_to_device(model, device=None):
    if(device == 'cuda'):
        if(torch.cuda.is_available()):
            device_count = torch.cuda.device_count()
            if device_count > 1:
                device_ids = list(range(device_count))
                # Automatically detect GPU device IDs
                model = nn.DataParallel(model, device_ids=device_ids)
                print(f"Using CUDA devices {device_ids}")
            else:
                print("Using CUDA device")
            return model.to('cuda')
        else:
            print("CUDA is not available")
    print("Using CPU")
    return model.to('cpu')


class PerceptualLoss(nn.Module):
    """Perceptual loss function"""
    def __init__(self):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg.children())[:36])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.criterion = nn.MSELoss()

        # Reshaped normalization parameters
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, output, target):
        # Normalize in compliance with ImageNet stats
        output = (output + 1) / 2  # Convert [-1, 1] to [0, 1] range
        target = (target + 1) / 2  # Convert [-1, 1] to [0, 1] range
        output = (output - self.mean.to(output.device)) / self.std.to(output.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)

        # Extract features
        output_features = self.feature_extractor(output)
        target_features = self.feature_extractor(target)

        # Compute perceptual loss
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
