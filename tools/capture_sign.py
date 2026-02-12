#!/usr/bin/env python3
#
# Capture tool: recognize ASL signs and take a photo when confirmed.
#
# Runs the MediaPipe gesture recognizer on the camera feed. When the
# accumulator confirms a letter (same sign detected for several consecutive
# frames), the raw camera frame is saved to images/captures/.
#
# Usage:
#   python tools/capture_sign.py                  # default camera (index 0)
#   python tools/capture_sign.py --camera 1       # camera index 1
#   python tools/capture_sign.py --camera /dev/video0
#   python tools/capture_sign.py --out ./my_pics  # custom output directory
#
# Keys:
#   R — reset accumulator
#   Q / ESC — quit
#

import sys
import os
import argparse
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import config
from recognizer_mediapipe import MediaPipeRecognizer

# Detection rate (seconds between detection runs)
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
    recognizer = MediaPipeRecognizer()

    print("\nASL Sign Capture")
    print("Show a sign to the camera — a photo is taken when the letter is confirmed.")
    print("R = reset, Q/ESC = quit\n")

    last_detect_time = 0
    last_annotated = None
    flash_until = 0          # timestamp until which the flash overlay is shown
    flash_letter = ""        # letter shown in the flash
    best_captures = {}       # letter -> confidence of saved image

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
                prev_conf = best_captures.get(confirmed, -1.0)
                if conf > prev_conf:
                    # Save the raw (un-annotated) frame, replacing any previous one
                    filename = f"{confirmed}.jpg"
                    filepath = os.path.join(args.out, filename)
                    cv2.imwrite(filepath, frame)
                    best_captures[confirmed] = conf

                    if prev_conf < 0:
                        print(f"  CAPTURED  sign '{confirmed}' (conf {conf:.2f})  ->  {filepath}")
                    else:
                        print(f"  REPLACED  sign '{confirmed}' (conf {prev_conf:.2f} -> {conf:.2f})  ->  {filepath}")

                    # Trigger flash overlay
                    flash_until = now + FLASH_DURATION
                    flash_letter = confirmed
                else:
                    print(f"  SKIPPED   sign '{confirmed}' (conf {conf:.2f} <= saved {prev_conf:.2f})")

                print(f"  Letters captured so far: {len(best_captures)}")

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
        elif key == ord('r'):
            recognizer.reset()
            print("  Accumulator reset")

    # --- Cleanup ---
    print(f"\nLetters captured: {len(best_captures)}")
    for letter, c in sorted(best_captures.items()):
        print(f"  {letter}: conf {c:.2f}")
    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
