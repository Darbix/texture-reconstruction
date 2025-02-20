import os
import cv2
import OpenEXR
import numpy as np
from glob import glob
import matplotlib.pyplot as plt


def load_data_iteratively(data_path, VIEW_IMGS_DIR, UV_IMGS_DIR, DATA_INFO_FILE,
    DATA_INFO_KEY):
    """Loads file paths for each data set scene by scene"""
    for scene in sorted(os.listdir(data_path)):
        # Path to UID specific set of data views
        scene_path = os.path.join(data_path, scene)
        if not os.path.isdir(scene_path):
            continue

        files = {
            VIEW_IMGS_DIR: sorted(glob(os.path.join(scene_path, VIEW_IMGS_DIR, "*.png"))),
            UV_IMGS_DIR:   sorted(glob(os.path.join(scene_path, UV_IMGS_DIR, "*.exr"))),
            DATA_INFO_KEY: os.path.join(scene_path, DATA_INFO_FILE) if os.path.exists(os.path.join(scene_path, DATA_INFO_FILE)) else None
        }

        yield scene, files


def save_img(aligned_img, img_path):
    """Saves an image and creates directories on the path"""
    dir_name = os.path.dirname(img_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(img_path, aligned_img)


def find_bounding_box(alpha_channel):
    """Finds the bounding box of nonzero alpha values"""
    non_zero_alpha = alpha_channel > 0.0
    if np.any(non_zero_alpha):
        min_y, min_x = np.min(np.nonzero(non_zero_alpha), axis=1)
        max_y, max_x = np.max(np.nonzero(non_zero_alpha), axis=1)
        return (min_x, min_y, max_x, max_y)
    else:
        return None


def crop_image_to_bbox(img, bounding_box):
    """Crops an image to the given bounding box"""
    if bounding_box:
        min_x, min_y, max_x, max_y = bounding_box
        cropped_img = img[min_y:max_y, min_x:max_x]
        return cropped_img
    else:
        return img


def load_exr_to_array(exr_filename):
    """Loads an EXR image into a NumPy array."""
    exr_file = OpenEXR.InputFile(exr_filename)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = exr_file.channels(["R", "G", "B", "A"])
    r, g, b, a = (np.frombuffer(ch, dtype=np.float16).reshape((height, width)) for ch in channels)
    return np.stack((r, g, b, a), axis=-1)


def resize_image(image_array, max_size, interpolation=cv2.INTER_LINEAR):
    """Resizes the numpy array image so that the longer side is max_size"""
    height, width = image_array.shape[:2]
    # Determine the current longer side
    longer_side = max(height, width)
    factor = max_size / longer_side
    # Resize the image and preserve aspect ratio
    resized_image = cv2.resize(image_array,
                               (int(width * factor), int(height * factor)),
                               interpolation=interpolation)
    return resized_image


def plot_image(image):
    """Plot a numpy image"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.max() > 1:
        image = image / 255.0

    plt.imshow(image)
    plt.show()
