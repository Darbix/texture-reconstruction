# config.py

from enum import Enum

# ----- Training ----- 
EPOCH_LOG_INTERVAL = 1  # Epoch logging interval
EPOCH_EVAL_INTERVAL = 1 # Epoch evaluation on val dataset interval

# ----- Paths -----
# Structure for particular training data scene directories is:
TEXTURE_PATCHES_DIR = "texture_patches" # Subdir with texture patches
VIEW_PATCHES_DIR = "image_patches"      # Subdir with view patches
REF_PATCHES_DIR = "reference_patches"   # Subdir with reference view patches
SCENE_INFO_DATA = "info.json"           # File with additional scene and patch data

class ModelType(Enum):
    RESNET34_UNET = 'RESNET34_UNET'
    MITB4_UNET = 'MITB4_UNET'
    WF_RESNET34_UNET = 'WF_RESNET34_UNET'
    MV_EDSR = 'MV_EDSR'
    RESNET101_UNETPLUSPLUS = 'RESNET101_UNETPLUSPLUS'
    