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
    
    # Resize back to the texture size
    if(uv_upsample_factor > 1.0):
        aligned_view_img = resize_image(aligned_view_img, max_size)
    
    # Enhance the image by inpainting the gaps due to low resolution UV mapping
    enhanced_img = inpaint_gaps(aligned_view_img)
    if(enhanced_img is not None):
        aligned_view_img = enhanced_img

    print("   Upscaled shape:", view_img_resized.shape,
          "Texture and output shape:", aligned_view_img.shape)
    
    return aligned_view_img


def inpaint_gaps(aligned_view_img):
    """Inpaint the missing UV mapped pixel causing gaps in an image"""
    # Find the transparent areas/gaps (alpha == 0)
    alpha_mask = aligned_view_img[:, :, 3] == 0

    # If there are some gaps or missing mapped pixels, impaint these areas
    if np.any(alpha_mask):
        # Separate the channels
        rgb_img = aligned_view_img[:, :, :3]
        # Convert boolean to mask (255 where alpha is 0)
        mask = alpha_mask.astype(np.uint8) * 255
        # Inverted mask (255 where the object has colors)
        inverted_mask = cv2.bitwise_not(mask)
        # Soften the edges
        blurred_mask = cv2.GaussianBlur(inverted_mask, (3, 3), 0)
        # Find the countours to get the total object mask
        contours, _ = cv2.findContours(blurred_mask,
                                       cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            # Create an empty mask for the largest contour
            object_mask = np.zeros_like(mask)
            # Fill the object mask
            cv2.drawContours(object_mask, [max_contour], -1, 255, thickness=cv2.FILLED)
            # Shrink the object bounding
            kernel = np.ones((3, 3), np.uint8)
            object_mask = cv2.erode(object_mask, kernel, iterations=1)

            # AND operation to combine mask for gaps with object mask
            # The background is ignored in inpainting
            mask = cv2.bitwise_and(mask, object_mask)

        # Inpainting the RGB channels with the mask and radius 3
        aligned_view_img[:, :, :3] = cv2.inpaint(rgb_img, mask, 3,
                                                 flags=cv2.INPAINT_TELEA)
        # Set alpha to 255 for the inpainted areas
        aligned_view_img[:, :, 3] = np.where(mask, 255, aligned_view_img[:, :, 3])

        return aligned_view_img

    else:
        return None


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
