#!/usr/bin/env python3
#
# Download or prepare ASL alphabet images for display.
#
# Option 1: Download from Kaggle (requires kaggle API credentials)
#   pip install kaggle
#   Set up ~/.kaggle/kaggle.json
#   python tools/download_images.py --kaggle
#
# Option 2: Generate placeholder images for testing
#   python tools/download_images.py --placeholders
#
# The Kaggle dataset is "grassknoted/asl-alphabet" which contains
# 200x200 photos of each ASL letter, matching the YOLO model's
# training distribution.
#

import os
import sys
import shutil
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

IMAGES_DIR = config.IMAGES_DIR


def generate_placeholders():
    """Generate simple placeholder images with hand-sign-like shapes for testing."""
    import cv2
    import numpy as np

    os.makedirs(IMAGES_DIR, exist_ok=True)
    size = 400

    for letter in config.LETTERS:
        img = np.zeros((size, size, 3), dtype=np.uint8)

        # Skin-tone background circle (simulates a hand)
        skin_color = (140, 180, 220)  # BGR, warm skin tone
        cv2.circle(img, (size // 2, size // 2), size // 3, skin_color, -1)

        # Large letter overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(letter, font, 5.0, 6)[0]
        x = (size - text_size[0]) // 2
        y = (size + text_size[1]) // 2
        cv2.putText(img, letter, (x, y), font, 5.0, (255, 255, 255), 6)

        path = os.path.join(IMAGES_DIR, f"{letter}.jpg")
        cv2.imwrite(path, img)

    print(f"Generated {len(config.LETTERS)} placeholder images in {IMAGES_DIR}/")
    print("NOTE: These are placeholders! For actual YOLO detection, replace with")
    print("real ASL hand photos from the Kaggle dataset.")


def download_kaggle():
    """Download ASL images from Kaggle dataset."""
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not installed. Install with:")
        print("  uv pip install kaggle")
        print("Then set up credentials: https://www.kaggle.com/docs/api")
        sys.exit(1)

    import tempfile

    dataset = "grassknoted/asl-alphabet"
    print(f"Downloading dataset: {dataset}")
    print("This is a large dataset (~1GB), but we only need one image per letter...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download just the test set (smaller)
        kaggle.api.dataset_download_files(dataset, path=tmpdir, unzip=True)

        # Look for the test or train images
        os.makedirs(IMAGES_DIR, exist_ok=True)

        for letter in config.LETTERS:
            # Try different possible paths in the dataset
            candidates = [
                os.path.join(tmpdir, "asl_alphabet_test", f"{letter}_test.jpg"),
                os.path.join(tmpdir, "asl_alphabet_train", letter),
            ]

            found = False
            for candidate in candidates:
                if os.path.isfile(candidate):
                    dest = os.path.join(IMAGES_DIR, f"{letter}.jpg")
                    shutil.copy2(candidate, dest)
                    found = True
                    break
                elif os.path.isdir(candidate):
                    # Pick the first image from the training folder
                    imgs = sorted(os.listdir(candidate))
                    if imgs:
                        src = os.path.join(candidate, imgs[0])
                        dest = os.path.join(IMAGES_DIR, f"{letter}.jpg")
                        shutil.copy2(src, dest)
                        found = True
                        break

            status = "OK" if found else "MISSING"
            print(f"  {letter}: {status}")

    print(f"\nImages saved to {IMAGES_DIR}/")


def check_images():
    """Check which images are present."""
    print(f"Image directory: {IMAGES_DIR}")
    present = []
    missing = []
    for letter in config.LETTERS:
        path_jpg = os.path.join(IMAGES_DIR, f"{letter}.jpg")
        path_png = os.path.join(IMAGES_DIR, f"{letter}.png")
        if os.path.exists(path_jpg) or os.path.exists(path_png):
            present.append(letter)
        else:
            missing.append(letter)

    print(f"  Present ({len(present)}): {' '.join(present)}")
    if missing:
        print(f"  Missing ({len(missing)}): {' '.join(missing)}")
    else:
        print("  All letters present!")


def main():
    parser = argparse.ArgumentParser(description="Download/prepare ASL images")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--kaggle", action="store_true",
                       help="Download from Kaggle dataset")
    group.add_argument("--placeholders", action="store_true",
                       help="Generate placeholder images for testing")
    group.add_argument("--check", action="store_true",
                       help="Check which images are present")
    args = parser.parse_args()

    if args.kaggle:
        download_kaggle()
    elif args.placeholders:
        generate_placeholders()
    elif args.check:
        check_images()


if __name__ == "__main__":
    main()
