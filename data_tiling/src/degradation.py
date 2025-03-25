# degradation.py

import cv2
import random
import numpy as np


def apply_strong_degradation(image):
    """Applies strong realistic degradation to an image"""
    degraded_image = image

    # Downsize image for processing
    degraded_image, original_size = downscale_image(
        degraded_image, factor=random.uniform(2.0, 2.7))
    
    degraded_image = apply_blur(
        degraded_image, kernel_size=random.choice([7, 9, 11]))
    degraded_image = apply_noise(
        degraded_image, stddev=random.randint(5, 12))
    dir = random.choice([(0, 1), (1, 0), (-1, 1), (1, -1), (-1, -1)])
    degraded_image = apply_chromatic_aberration(
        degraded_image, shift=random.randint(0, 5), dir_h=dir[0], dir_v=dir[1])
    degraded_image = apply_motion_blur(
        degraded_image, kernel_size=random.choice([0, 1, 3]))
    degraded_image, alpha_channel = apply_cfa(degraded_image)
    degraded_image = ahd_demosaic(degraded_image, alpha_channel=alpha_channel)
    degraded_image = apply_jpeg_compression(
        degraded_image, quality=random.randint(25, 65))
    
    # Upscale back to original size
    degraded_image = upscale_image(degraded_image, original_size)

    if degraded_image.shape[2] == 4:
        alpha_mask = degraded_image[:, :, 3] == 0
        # Set RGB channels to 0 where alpha is 0
        degraded_image[alpha_mask, :3] = 0

    return degraded_image


def apply_light_degradation(image):
    """Applies small realistic degradation to an image"""
    degraded_image = image

    degraded_image = apply_noise(
        degraded_image, stddev=random.uniform(0, 1.8))
    dir = random.choice([(0, 1), (1, 0), (-1, 1), (1, -1), (-1, -1)])
    degraded_image = apply_chromatic_aberration(
        degraded_image, shift=random.randint(0, 3), dir_h=dir[0], dir_v=dir[1])
    degraded_image, alpha_channel = apply_cfa(degraded_image)
    degraded_image = ahd_demosaic(degraded_image, alpha_channel=alpha_channel)
    degraded_image = apply_jpeg_compression(
        degraded_image, quality=random.randint(70, 100))
    mean, stddev, max_shift = 0, 1.8, 5
    degraded_image = apply_displacement(
        degraded_image,
        dx=np.clip(np.random.normal(mean, stddev), -max_shift, max_shift),
        dy=np.clip(np.random.normal(mean, stddev), -max_shift, max_shift))

    if degraded_image.shape[2] == 4:
        alpha_mask = degraded_image[:, :, 3] == 0
        # Set RGB channels to 0 where alpha is 0
        degraded_image[alpha_mask, :3] = 0

    return degraded_image


def downscale_image(image, factor=2.0):
    """Downscales an image by a scaling factor"""
    h, w = image.shape[:2]
    image = cv2.resize(image, (int(w / factor), int(h / factor)), interpolation=cv2.INTER_AREA)
    return image, (w, h)


def upscale_image(image, original_size):
    """Upscales an image by a scaling factor"""
    return cv2.resize(image, original_size, interpolation=cv2.INTER_CUBIC)


def apply_blur(image, kernel_size=9):
    """Applies blur to an image with a specific kernel size"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_noise(image, mean=0, stddev=10):
    """Applies noise to an image as normal distributed values"""
    noise = np.random.normal(mean, stddev, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image


def apply_chromatic_aberration(image, shift=3, dir_h=1, dir_v=0):
    """
    Applies chromatic aberration to an image with a specific shift size and
    in a specific direction
    """
    if shift == 0:
        return image

    b, g, r = cv2.split(image[:, :, :3])  # Extract RGB channels
    
    # horizontal
    if(dir_h != 0):
        r = np.roll(r, shift * dir_h, axis=1)   # Shift R channel by 'shift' pixels
        b = np.roll(b, -shift * dir_h, axis=1)  # Shift B channel by 'shift' pixels
    # Vertical
    if(dir_v != 0):
        r = np.roll(r, shift * dir_v, axis=0)   # Shift R channel by 'shift' pixels
        b = np.roll(b, -shift * dir_v, axis=0)  # Shift B channel by 'shift' pixels

    aberrated_image = cv2.merge([b, g, r])

    # If image has an alpha channel, preserve it
    if image.shape[2] == 4:
        a = image[:, :, 3]
        aberrated_image = cv2.merge([b, g, r, a])
    
    return aberrated_image


def apply_motion_blur(image, kernel_size=3):
    """Applies motion blur to an image with a specific kernel size"""
    if(kernel_size == 0):
        return image
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    return cv2.filter2D(image, -1, kernel)


def apply_cfa(image):
    """Applies Color Filter Array (CFA) to an image"""
    # Check if image has an alpha channel and extract it
    has_alpha = image.shape[-1] == 4
    if has_alpha:
        alpha_channel = image[:, :, 3]
    
    # Simulating Bayer RGGB CFA (Color Filter Array)
    bayer_pattern = np.zeros_like(image[:, :, 0], dtype=image.dtype)
    bayer_pattern[0::2, 0::2] = image[0::2, 0::2, 2]  # Red (even elements)
    bayer_pattern[1::2, 1::2] = image[1::2, 1::2, 0]  # Blue (odd elements)
    bayer_pattern[0::2, 1::2] = image[0::2, 1::2, 1]  # Green (even rows, odd columns)
    bayer_pattern[1::2, 0::2] = image[1::2, 0::2, 1]  # Green (odd rows, even columns)

    # If image had an alpha channel, return both the Bayer pattern and alpha
    if has_alpha:
        return bayer_pattern, alpha_channel
    return bayer_pattern, None


def ahd_demosaic(bayer_image, alpha_channel=None):
    """
    Applies Adaptive Homogeneity-Directed interpolation for demosaicing
    a Bayer-patterned image
    """
    # Interpolation of Bayer mosaic image
    demosaiced_image = cv2.cvtColor(bayer_image, cv2.COLOR_BAYER_BG2BGR)

    # Reattach alpha channel if available
    if alpha_channel is not None:
        return np.dstack((demosaiced_image, alpha_channel))
    return demosaiced_image


def apply_jpeg_compression(image, quality=30):
    """Applies JPEG compression to an image to degrade the quality"""
    # Check if the image has an alpha channel
    has_alpha = image.shape[-1] == 4
    if has_alpha:
        alpha_channel = image[:, :, 3]  # Extract alpha channel
        image = image[:, :, :3]  # Remove alpha for JPEG compression

    # Apply JPEG compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded_image = cv2.imencode('.jpg', image, encode_param)
    decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)

    # Reattach alpha channel if it was present
    if has_alpha:
        decoded_image = np.dstack((decoded_image, alpha_channel))

    return decoded_image


def apply_displacement(image, dx=0, dy=0):
    """Applies pixel displacement to simulate wrong optical flow alignment"""
    height, width = image.shape[:2]

    # Apply the displacement to the image
    map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = map_x + dx
    map_y = map_y + dy

    # Clip values to ensure they are within the bounds of the image
    map_x = np.clip(map_x, 0, width - 1)
    map_y = np.clip(map_y, 0, height - 1)

    # Remap the image to get the displaced image
    image = cv2.remap(
        image,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR
    )
    return image
