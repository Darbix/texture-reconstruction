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

# MVTRN types of models
class ModelType(Enum):
    DEFAULT = 'DEFAULT'
    UNET = 'UNET'
