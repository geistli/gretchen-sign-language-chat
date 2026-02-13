#!/usr/bin/env python3
#
# Test tool: detect ASL letters and build sentences using MediaPipe.
#
# Drop your hand briefly between words to insert a space.
#
# Usage:
#   python tools/test_recognizer_mediapipe.py                # Default camera
#   python tools/test_recognizer_mediapipe.py --camera 0     # Webcam index 0
#
# Keys:
#   C — clear sentence
#   BACKSPACE — delete last letter
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

# Frames with no hand detected before inserting a space
SPACE_GAP_FRAMES = 8


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

    print("MediaPipe ASL Sentence Recognizer")
    print("Sign letters — drop your hand between words for a space")
    print("C=clear, BACKSPACE=delete last, Q/ESC=quit")
    print()

    sentence = []          # list of characters (letters and spaces)
    no_hand_frames = 0     # frames since last hand detection
    space_inserted = True  # prevent multiple spaces in a row
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

            # Track whether a hand is visible
            if best is not None:
                no_hand_frames = 0
                space_inserted = False
            else:
                no_hand_frames += 1

            # Insert space when hand has been gone long enough
            if no_hand_frames >= SPACE_GAP_FRAMES and not space_inserted and sentence and sentence[-1] != " ":
                sentence.append(" ")
                space_inserted = True
                print("  [SPACE]")

            if confirmed:
                # Skip duplicate consecutive letters
                if not sentence or sentence[-1] != confirmed:
                    sentence.append(confirmed)
                    text = "".join(sentence)
                    print(f"  [{confirmed}]  sentence: {text}")

        # Draw sentence on the frame
        display = last_annotated if last_annotated is not None else frame
        text = "".join(sentence)

        # Draw background bar for sentence
        h, w = display.shape[:2]
        cv2.rectangle(display, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(display, text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("MediaPipe ASL", display)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
        elif key == ord('c'):
            sentence.clear()
            recognizer.clear()
            space_inserted = True
            print("  Cleared.")
        elif key == 8 or key == 127:  # backspace
            if sentence:
                removed = sentence.pop()
                print(f"  Deleted: '{removed}'")

    if sentence:
        text = "".join(sentence)
        print(f"\nFinal sentence: {text}")

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
