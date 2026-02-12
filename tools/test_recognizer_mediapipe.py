#!/usr/bin/env python3
#
# Test tool: detect ASL letters using MediaPipe gesture recognizer.
#
# Usage:
#   python tools/test_recognizer_mediapipe.py                # Default camera
#   python tools/test_recognizer_mediapipe.py --camera 0     # Webcam index 0
#
# Keys:
#   R — reset accumulator
#   Q/ESC — quit
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

# Run detection this many times per second
DETECT_INTERVAL = 1.0 / 5.0


def main():
    parser = argparse.ArgumentParser(description="Test MediaPipe ASL recognizer")
    parser.add_argument("--camera", default=config.CAMERA_DEV,
                        help="Camera device path or index (default: /dev/grt_cam)")
    args = parser.parse_args()

    # Open camera
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

    # Load recognizer
    recognizer = MediaPipeRecognizer()

    print("MediaPipe ASL Recognizer Test")
    print("R=reset, Q/ESC=quit")

    last_detect_time = 0
    last_annotated = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()

        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now
            confirmed, best, conf, annotated = recognizer.process_frame(frame)
            last_annotated = annotated

            if confirmed:
                print(f"  CONFIRMED: {confirmed}")

        if last_annotated is not None:
            cv2.imshow("MediaPipe ASL", last_annotated)
        else:
            cv2.imshow("MediaPipe ASL", frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            recognizer.reset()
            print("  Accumulator reset")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
