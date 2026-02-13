#!/usr/bin/env python3
"""
Live ASL Chat — Claude Code is directly in the loop.

The recognizer detects letters. When you press SPACE, the word is written
to /tmp/asl_input. Claude Code reads it, decides a response, and writes
it to /tmp/asl_response. The display shows the response as ASL images.

Usage:
    python chat_live.py --camera 0

Keys:
    SPACE  — send current word (writes to /tmp/asl_input)
    C      — clear current word
    Q/ESC  — quit (writes QUIT to /tmp/asl_input)
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import config
from recognizer_mediapipe import MediaPipeRecognizer
from display import ASLDisplay

DETECT_INTERVAL = 1.0 / 5.0

INPUT_FILE = "/tmp/asl_input"
RESPONSE_FILE = "/tmp/asl_response"

# Use own images for display
OWN_IMAGES_DIR = os.path.join(config.BASE_DIR, "images", "own")


def cleanup():
    """Remove communication files."""
    for f in (INPUT_FILE, RESPONSE_FILE):
        if os.path.exists(f):
            os.remove(f)


def wait_for_response(display, timeout=60):
    """Poll for response file, keeping the display alive."""
    start = time.time()
    while time.time() - start < timeout:
        if os.path.exists(RESPONSE_FILE):
            with open(RESPONSE_FILE) as f:
                response = f.read().strip().upper()
            os.remove(RESPONSE_FILE)
            return response if response else None
        # Keep the cv2 event loop alive
        key = cv2.waitKey(100)
        if key == 27 or key == ord("q"):
            return None
    print("  (timed out waiting for Claude)")
    return None


def display_response(display, word):
    """Show response letter by letter, then the full word."""
    print(f"  Gretchen says: {word}")

    for letter in word:
        display.show_letter(letter, border_color=config.COLOR_GREEN)
        start = time.time()
        while time.time() - start < config.LETTER_DISPLAY_TIME:
            key = cv2.waitKey(30)
            if key == 27 or key == ord("q"):
                return False

        display.show_blank(border_color=config.COLOR_GREEN)
        start = time.time()
        while time.time() - start < config.LETTER_PAUSE_TIME:
            cv2.waitKey(30)

    # Show full word
    display.show_word(word, border_color=config.COLOR_CYAN)
    start = time.time()
    while time.time() - start < 2.0:
        key = cv2.waitKey(30)
        if key == 27 or key == ord("q"):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Live ASL Chat with Claude Code")
    parser.add_argument(
        "--camera", default=config.CAMERA_DEV,
        help="Camera device path or index",
    )
    parser.add_argument("--fullscreen", action="store_true")
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

    # Point display at own/ images
    config.IMAGES_DIR = OWN_IMAGES_DIR

    recognizer = MediaPipeRecognizer()
    display = ASLDisplay(fullscreen=args.fullscreen)

    # Clean start
    cleanup()

    display.show_blank(border_color=config.COLOR_CYAN)

    print()
    print("=== Live ASL Chat ===")
    print("Claude Code is in the loop — responses come from your terminal session.")
    print("SPACE = send word | C = clear | Q/ESC = quit")
    print()

    # Check if Claude wants to speak first (response file already there)
    if os.path.exists(RESPONSE_FILE):
        response = wait_for_response(display)
        if response:
            display_response(display, response)
            display.show_blank(border_color=config.COLOR_CYAN)

    history = []
    last_detect_time = 0

    while True:
        # Check if Claude sent a response proactively
        if os.path.exists(RESPONSE_FILE):
            with open(RESPONSE_FILE) as f:
                response = f.read().strip().upper()
            os.remove(RESPONSE_FILE)
            if response:
                valid = "".join(c for c in response if c in config.LETTERS)
                if valid:
                    history.append(("sent", valid))
                    if not display_response(display, valid):
                        break
                    display.show_blank(border_color=config.COLOR_CYAN)

        ok, frame = cap.read()
        if not ok:
            continue

        now = time.time()
        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now
            confirmed, _, _, annotated = recognizer.process_frame(frame)

            if confirmed:
                word_so_far = "".join(recognizer.word_buffer)
                print(f"  [{confirmed}]  word: {word_so_far}")
        else:
            annotated = frame

        # Draw conversation history
        y = annotated.shape[0] - 20
        for direction, w in reversed(history[-6:]):
            label = f"{'You' if direction == 'received' else 'Gretchen'}: {w}"
            color = (200, 200, 200) if direction == "received" else (0, 255, 200)
            cv2.putText(annotated, label, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y -= 25

        cv2.imshow("ASL Camera", annotated)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            with open(INPUT_FILE, "w") as f:
                f.write("QUIT")
            break

        elif key == ord("c"):
            cleared = recognizer.get_word()
            if cleared:
                print(f"  Cleared: {cleared}")

        elif key == ord(" "):
            word = recognizer.get_word()
            if not word:
                print("  (no letters yet)")
                continue

            print(f"\n  You signed: {word}")
            history.append(("received", word))

            # Write to file for Claude Code to read
            with open(INPUT_FILE, "w") as f:
                f.write(word)
            print("  Waiting for Claude...")

            # Show waiting state
            display.show_word("...", border_color=config.COLOR_CYAN)

            # Wait for response
            response = wait_for_response(display)
            if response is None:
                print("  (no response)")
                display.show_blank(border_color=config.COLOR_CYAN)
                continue

            valid = "".join(c for c in response if c in config.LETTERS)
            if not valid:
                print("  (invalid response)")
                display.show_blank(border_color=config.COLOR_CYAN)
                continue

            history.append(("sent", valid))

            if not display_response(display, valid):
                break

            display.show_blank(border_color=config.COLOR_CYAN)
            print()

    # Summary
    if history:
        print("\n=== Conversation ===")
        for direction, w in history:
            speaker = "You" if direction == "received" else "Gretchen"
            print(f"  {speaker}: {w}")

    cap.release()
    display.close()
    cv2.destroyAllWindows()
    cleanup()
    print("Done.")


if __name__ == "__main__":
    main()
