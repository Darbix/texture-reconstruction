import cv2
import numpy as np

from image_utils import load_exr_to_array, find_bounding_box,\
    crop_image_to_bbox, resize_image


def align_image_uv(texture_shape, view_img_path, uv_img_path, uv_upscale_factor=1.5):
    """Aligns the input image using UV mapping"""
    # Load the view image
    view_img = cv2.imread(view_img_path, cv2.IMREAD_UNCHANGED)
    # Load the UV image
    uv_img = load_exr_to_array(uv_img_path)
    # Get the reference texture dimensions
    texture_h, texture_w, _ = texture_shape
    max_size = max(texture_h, texture_w)

    alpha_channel = uv_img[:, :, -1]
    bbox = find_bounding_box(alpha_channel)

    view_img_cropped = crop_image_to_bbox(view_img, bbox)
    uv_img_cropped = crop_image_to_bbox(uv_img, bbox)

    # The input UV map has much smaller resolution than the texture and there
    # might not be all corresponding mapping due to unbalanced surface density.
    # Factor resizes (interpolates) the UV map and the image to get more precise
    # mapping 'image pixel to texture pixel'. Then it's resized back to max_size
    uv_upsample_factor = uv_upscale_factor
    max_size_upsampled = int(max_size * uv_upsample_factor)

    view_img_resized = resize_image(view_img_cropped, max_size_upsampled)
    uv_img_resized = resize_image(uv_img_cropped.astype(np.float32), max_size_upsampled)

    # Visualization of the cropped and resized UV map
    # plt.imshow((uv_img_resized * 255).astype(np.uint8))
    # plt.show()

    aligned_view_img = remap_texture(view_img_resized, uv_img_resized,
                                     texture_h, texture_w)

    aligned_view_img = resize_image(aligned_view_img, max_size)

    print("   Upscaled shape:", view_img_resized.shape,
          "Texture and output shape:", aligned_view_img.shape)
    return aligned_view_img


def remap_texture(png_array, exr_array, ref_h, ref_w):
    """Remaps a PNG texture using UV coordinates from an EXR image"""
    output_image = np.zeros((ref_h, ref_w, 4), dtype=np.uint8)
    # Alpha channel serves as a mask to where the object is
    alpha_channel = exr_array[:, :, -1]
    mask = alpha_channel > 0.0
    # From RGBA the R channel is a float value for U and the G is for V
    u = (exr_array[:, :, 0] * (ref_w - 1)).astype(int)
    v = ((1.0 - exr_array[:, :, 1]) * (ref_h - 1)).astype(int)

    # Mask valid UV values only in the surface area
    valid = (u >= 0) & (u < ref_w) & (v >= 0) & (v < ref_h) & mask

    output_image[v[valid], u[valid]] = png_array[valid]

    return output_image
