#!/usr/bin/env python3
#
# Capture tool: recognize ASL signs and take a photo when confirmed.
#
# Runs the YOLO model on the camera feed. When the accumulator confirms
# a letter (same sign detected for several consecutive frames), the raw
# camera frame is saved to images/captures/.
#
# Usage:
#   python tools/capture_sign.py                  # default camera (index 0)
#   python tools/capture_sign.py --camera 1       # camera index 1
#   python tools/capture_sign.py --camera /dev/video0
#   python tools/capture_sign.py --out ./my_pics  # custom output directory
#
# Keys:
#   C — clear accumulated word
#   Q / ESC — quit
#

import sys
import os
import argparse
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import config
from recognizer import ASLRecognizer

# Detection rate (seconds between YOLO runs)
DETECT_INTERVAL = 1.0 / 5.0

# How long to show the "CAPTURED!" flash overlay (seconds)
FLASH_DURATION = 0.6


def main():
    parser = argparse.ArgumentParser(
        description="Recognize ASL signs and capture a photo on confirmation",
    )
    parser.add_argument(
        "--camera", default="0",
        help="Camera device path or index (default: 0)",
    )
    parser.add_argument(
        "--out", default=os.path.join(config.IMAGES_DIR, "own"),
        help="Directory to save captured photos (default: images/own/)",
    )
    args = parser.parse_args()

    # --- Open camera ---
    cam = args.camera
    try:
        cam = int(cam)
    except ValueError:
        pass
    cap = cv2.VideoCapture(cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    if not cap.isOpened():
        print(f"Error: cannot open camera {cam}")
        sys.exit(1)

    # --- Prepare output directory ---
    os.makedirs(args.out, exist_ok=True)
    print(f"Photos will be saved to: {os.path.abspath(args.out)}")

    # --- Load recognizer ---
    recognizer = ASLRecognizer()

    print("\nASL Sign Capture")
    print("Show a sign to the camera — a photo is taken when the sign is confirmed.")
    print("C = clear word, Q/ESC = quit\n")

    last_detect_time = 0
    last_annotated = None
    flash_until = 0          # timestamp until which the flash overlay is shown
    flash_letter = ""        # letter shown in the flash
    capture_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()

        # --- Run detection at fixed interval ---
        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now
            confirmed, best, conf, annotated = recognizer.process_frame(frame)
            last_annotated = annotated

            if confirmed:
                # Save the raw (un-annotated) frame
                capture_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{confirmed}_{timestamp}_{capture_count}.jpg"
                filepath = os.path.join(args.out, filename)
                cv2.imwrite(filepath, frame)

                print(f"  CAPTURED  sign '{confirmed}'  ->  {filepath}")
                print(f"  Word so far: {''.join(recognizer.word_buffer)}")

                # Trigger flash overlay
                flash_until = now + FLASH_DURATION
                flash_letter = confirmed

        # --- Build display frame ---
        display = last_annotated if last_annotated is not None else frame

        # Flash overlay when a photo was just taken
        if now < flash_until:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0),
                          (display.shape[1], display.shape[0]),
                          (0, 255, 0), -1)
            # Blend: 30 % green flash
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

            text = f"CAPTURED: {flash_letter}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            tx = (display.shape[1] - text_size[0]) // 2
            ty = display.shape[0] - 30
            cv2.putText(display, text, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("ASL Sign Capture", display)

        # --- Keyboard handling ---
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        elif key == ord('c'):
            word = recognizer.get_word()
            print(f"  Word cleared: {word}")

    # --- Cleanup ---
    word = recognizer.get_word()
    if word:
        print(f"\nFinal word: {word}")

    print(f"\nTotal photos captured: {capture_count}")
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
