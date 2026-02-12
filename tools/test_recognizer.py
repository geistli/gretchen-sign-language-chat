#!/usr/bin/env python3
#
# Test tool: detect ASL letters from camera feed.
#
# Usage:
#   python tools/test_recognizer.py             # Use default camera
#   python tools/test_recognizer.py --camera 1  # Use camera index 1
#
# Keys:
#   C — clear accumulated word
#   B — toggle border color detection display
#   Q/ESC — quit
#

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import cv2
import config
from recognizer import ASLRecognizer, detect_border_color

# Run YOLO detection this times per second, show camera feed in between
DETECT_INTERVAL = 1.0 / 5.0
1

def main():
    parser = argparse.ArgumentParser(description="Test ASL recognizer")
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
    recognizer = ASLRecognizer()

    print("ASL Recognizer Test")
    print("C=clear word, B=show border detection, Q/ESC=quit")

    show_border = False
    last_detect_time = 0
    last_annotated = None

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()

        # Run YOLO only every DETECT_INTERVAL seconds
        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now
            confirmed, best, conf, annotated = recognizer.process_frame(frame)
            last_annotated = annotated

            if confirmed:
                print(f"  CONFIRMED: {confirmed}  (word so far: {''.join(recognizer.word_buffer)})")

            if show_border:
                border = detect_border_color(frame)
                color_text = f"Border: {border or 'none'}"
                cv2.putText(
                    annotated, color_text, (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2,
                )
        # Between detections, keep showing the last annotated frame
        # so detection boxes don't flicker on/off
        if last_annotated is not None:
            cv2.imshow(config.CAMERA_WINDOW, last_annotated)
        else:
            cv2.imshow(config.CAMERA_WINDOW, frame)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        elif key == ord('c'):
            word = recognizer.get_word()
            print(f"  Word cleared: {word}")
        elif key == ord('b'):
            show_border = not show_border
            print(f"  Border detection: {'ON' if show_border else 'OFF'}")

    word = recognizer.get_word()
    if word:
        print(f"\nFinal word: {word}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
