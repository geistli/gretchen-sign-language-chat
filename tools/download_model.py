#!/usr/bin/env python3
#
# Download the pretrained ASL YOLO model from HuggingFace.
#
# Model: atalaydenknalbant/asl-yolo-models (YOLOv8s trained on ASL alphabet)
#
# Usage:
#   python tools/download_model.py
#

import os
import sys
import urllib.request

# HuggingFace model URL
REPO = "atalaydenknalbant/asl-yolo-models"
FILENAME = "yolov8s.pt"
URL = f"https://huggingface.co/{REPO}/resolve/main/{FILENAME}"

# Destination
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEST_PATH = os.path.join(MODEL_DIR, "yolov8s_asl.pt")


def download():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(DEST_PATH):
        size_mb = os.path.getsize(DEST_PATH) / (1024 * 1024)
        print(f"Model already exists at {DEST_PATH} ({size_mb:.1f} MB)")
        resp = input("Re-download? [y/N] ").strip().lower()
        if resp != "y":
            return

    print(f"Downloading from: {URL}")
    print(f"Saving to: {DEST_PATH}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f} / {total_mb:.1f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(URL, DEST_PATH, reporthook=progress)
        print()
        size_mb = os.path.getsize(DEST_PATH) / (1024 * 1024)
        print(f"Done! Model saved ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"\nError downloading model: {e}")
        print(f"\nYou can manually download from:")
        print(f"  {URL}")
        print(f"and save it as:")
        print(f"  {DEST_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    download()
