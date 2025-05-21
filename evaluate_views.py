# evaluate_views.py

import os
import cv2
import argparse
import datetime
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor as Executor

np.seterr(divide='ignore', invalid='ignore') # Ignores edge PSNR values


def extract_patches(image_np, patch_size, stride):
    patches = []
    positions = []

    height, width = image_np.shape[:2]
    pad_right = (stride - (width - patch_size) % stride) % stride
    pad_bottom = (stride - (height - patch_size) % stride) % stride

    if pad_right > 0 or pad_bottom > 0:
        image_np = cv2.copyMakeBorder(image_np, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        height, width = image_np.shape[:2]

    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = image_np[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
            positions.append((x, y))

    return patches, positions, image_np


def apply_color_transfer(source, target):
    """Color transfer in LAB space for color harmonization"""
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    
    mean_src, std_src = cv2.meanStdDev(source_lab)
    mean_trg, std_trg = cv2.meanStdDev(target_lab)

    mean_src = mean_src.reshape(1, 1, 3)
    std_src = std_src.reshape(1, 1, 3)
    mean_trg = mean_trg.reshape(1, 1, 3)
    std_trg = std_trg.reshape(1, 1, 3)
    
    result = (target_lab - mean_trg) * (std_src / (std_trg + 1e-6)) + mean_src
    result = np.clip(result, 0, 255).astype("uint8")

    return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def compare_patch(args):
    texture_patch, img_patch, x, y, patch_size, patch_stride = args
    
    # more than 60% of a patch is black (pixel level < 12) => skip
    if (cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY) < 12).mean() > 0.60:
        return None  # Skip dark patches

    img_patch = apply_color_transfer(cv2.cvtColor(texture_patch, cv2.COLOR_RGB2BGR), img_patch)
    img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)

    psnr_score = psnr(texture_patch, img_patch)
    ssim_score = ssim(texture_patch, img_patch, channel_axis=-1)

    pos_y, pos_x = (y // patch_stride, x // patch_stride)

    return (pos_y, pos_x, img_patch, psnr_score, ssim_score)


def compare_patch_batch(batch):
    batch_results = []
    for args in batch:
        result = compare_patch(args)
        if result is not None:
            batch_results.append(result)
    return batch_results


def main(texture_path, images_dir, patch_size, patch_stride, output_dir):
    """
    Compares each patch from a texture to all view patches at the same position and selects
    the one with the best PSNR and one with the best SSIM. The final metrics are evaluated
    on whole composed images. The processing is parallelized.
    """
    texture_img = cv2.cvtColor(cv2.imread(texture_path), cv2.COLOR_BGR2RGB)
    height, width = texture_img.shape[:2]

    texture_patches, tex_positions, padded_texture_img = extract_patches(texture_img, patch_size, patch_stride)
    padded_height, padded_width = padded_texture_img.shape[:2]

    best_patches_psnr = [[(-1, -1) for _ in range(0, padded_width - patch_size + 1, patch_stride)]
                         for _ in range(0, padded_height - patch_size + 1, patch_stride)]
    best_patches_ssim = [[(-1, -1) for _ in range(0, padded_width - patch_size + 1, patch_stride)]
                         for _ in range(0, padded_height - patch_size + 1, patch_stride)]

    composed_image_psnr = np.zeros_like(padded_texture_img)
    composed_image_ssim = np.zeros_like(padded_texture_img)

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

    for img_id, img_file in enumerate(image_files):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)

        img_patches, _, _ = extract_patches(img, patch_size, patch_stride)

        print(f"Image {img_id} {img_file} loaded")

        tasks = [(texture_patches[i], img_patches[i], x, y, patch_size, patch_stride) 
                 for i, (x, y) in enumerate(tex_positions)]

        batch_size = 512
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

        with Executor(max_workers=os.cpu_count()) as executor:
            batch_results = executor.map(compare_patch_batch, batches)

            for batch in batch_results:
                for result in batch:
                    pos_y, pos_x, img_patch, psnr_score, ssim_score = result

                    best_id_psnr, best_psnr = best_patches_psnr[pos_y][pos_x]
                    best_id_ssim, best_ssim = best_patches_ssim[pos_y][pos_x]

                    if psnr_score > best_psnr:
                        best_patches_psnr[pos_y][pos_x] = (img_id, psnr_score)
                        composed_image_psnr[pos_y * patch_stride: pos_y * patch_stride + patch_size,
                                            pos_x * patch_stride: pos_x * patch_stride + patch_size] = img_patch

                    if ssim_score > best_ssim:
                        best_patches_ssim[pos_y][pos_x] = (img_id, ssim_score)
                        composed_image_ssim[pos_y * patch_stride: pos_y * patch_stride + patch_size,
                                            pos_x * patch_stride: pos_x * patch_stride + patch_size] = img_patch

    composed_image_psnr = composed_image_psnr[0:height, 0:width]
    composed_image_ssim = composed_image_ssim[0:height, 0:width]

    print(f"composed_image_psnr PSNR: {psnr(texture_img, composed_image_psnr)}")
    print(f"composed_image_ssim SSIM: {ssim(texture_img, composed_image_ssim, channel_axis=-1)}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist
    
        current_date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        cv2.imwrite(os.path.join(output_dir, f'composed_image_psnr_{current_date}.jpg'),
                    cv2.cvtColor(composed_image_psnr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir, f'composed_image_ssim_{current_date}.jpg'),
                    cv2.cvtColor(composed_image_ssim, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Patch Comparison")
    parser.add_argument('--texture_image', type=str, required=True, help='Path to the texture image')
    parser.add_argument('--images_dir', type=str, required=True, help='Directory of images for comparison')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory for output images')
    parser.add_argument('--patch_size', type=int, default=32, help='Patch size for the processing')
    parser.add_argument('--patch_stride', type=int, help='Patch stride')
    args = parser.parse_args()

    if(not args.patch_stride):
        args.patch_stride = args.patch_size

    main(args.texture_image, args.images_dir, args.patch_size, args.patch_stride, args.output_dir)
