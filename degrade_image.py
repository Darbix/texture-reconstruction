# degrade_image.py

import argparse
import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'data_tiling', 'src'))

from degradation import apply_strong_degradation, set_cfg

def main():
    parser = argparse.ArgumentParser(description='Applies degradation to an image.')
    parser.add_argument('--input_image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_image', type=str, required=True, help='Path to save degraded image')
    parser.add_argument("--degradation_config", type=str, help="Path to YAML degradation config file")
    args = parser.parse_args()

    set_cfg(args.degradation_config)

    image = cv2.imread(args.input_image)
    if image is None:
        raise FileNotFoundError(f"Cannot load image at {args.input_image}")

    image = apply_strong_degradation(image)

    cv2.imwrite(args.output_image, image)
    print(f"Degraded image saved to {args.output_image}")

if __name__ == '__main__':
    main()
