#!/usr/bin/env python3
"""
Chat with Claude via ASL Sign Language.

You sign letters to the camera → MediaPipe recognizes them →
Claude thinks of a response → the response is displayed as ASL images.

Usage:
    python chat_with_claude.py                # default camera
    python chat_with_claude.py --camera 0     # webcam index

Keys:
    SPACE  — send current word to Claude
    C      — clear current word
    Q/ESC  — quit
"""

import sys
import os
import argparse
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import config
from recognizer_mediapipe import MediaPipeRecognizer
from display import ASLDisplay

DETECT_INTERVAL = 1.0 / 5.0

# Use own images for display
OWN_IMAGES_DIR = os.path.join(config.BASE_DIR, "images", "own")


def ask_claude(word, history):
    """Ask Claude for a response via the claude CLI."""
    history_lines = []
    for direction, w in history:
        speaker = "Human" if direction == "received" else "Gretchen"
        history_lines.append(f"  {speaker}: {w}")
    history_text = "\n".join(history_lines) if history_lines else "  (conversation just started)"

    prompt = (
        f"You are Gretchen, a small humanoid robot chatting in sign language. "
        f"Someone just signed the word \"{word}\" to you.\n\n"
        f"Conversation so far:\n{history_text}\n  Human: {word}\n\n"
        f"Reply with ONE short word (1-8 letters) using ONLY these letters: "
        f"A B C D E F G H I K L M N O P Q R S T U V W X Y\n"
        f"(No J or Z — those need motion in ASL.)\n\n"
        f"Just the word, nothing else. No punctuation, no explanation."
    )

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=30,
        )
        response = result.stdout.strip().upper()
        # Keep only valid ASL letters
        filtered = "".join(c for c in response if c in config.LETTERS)
        return filtered[:8] if filtered else "HI"
    except subprocess.TimeoutExpired:
        print("  (Claude took too long, defaulting to HI)")
        return "HI"
    except FileNotFoundError:
        print("  (claude CLI not found — install Claude Code first)")
        return "HI"
    except Exception as e:
        print(f"  (Claude error: {e})")
        return "HI"


def display_response(display, word):
    """Show Claude's response letter by letter, then the full word."""
    print(f"  Gretchen says: {word}")

    for letter in word:
        display.show_letter(letter, border_color=config.COLOR_GREEN)
        # Pump the event loop so the window actually updates
        start = time.time()
        while time.time() - start < config.LETTER_DISPLAY_TIME:
            key = cv2.waitKey(30)
            if key == 27 or key == ord("q"):
                return False

        # Brief pause between letters
        display.show_blank(border_color=config.COLOR_GREEN)
        start = time.time()
        while time.time() - start < config.LETTER_PAUSE_TIME:
            cv2.waitKey(30)

    # Show full word for a moment
    display.show_word(word, border_color=config.COLOR_CYAN)
    start = time.time()
    while time.time() - start < 2.0:
        key = cv2.waitKey(30)
        if key == 27 or key == ord("q"):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Chat with Claude via ASL")
    parser.add_argument(
        "--camera", default=config.CAMERA_DEV,
        help="Camera device path or index (default: /dev/grt_cam)",
    )
    parser.add_argument("--fullscreen", action="store_true", help="Fullscreen display")
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

    # Init recognizer and display
    recognizer = MediaPipeRecognizer()
    display = ASLDisplay(fullscreen=args.fullscreen)

    # Show ready state
    display.show_blank(border_color=config.COLOR_CYAN)

    print()
    print("=== Chat with Claude via ASL ===")
    print("Sign letters to the camera.")
    print("SPACE = send word to Claude | C = clear | Q/ESC = quit")
    print()

    history = []  # list of (direction, word)
    last_detect_time = 0

    while True:
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

        # Draw conversation history on camera view
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
            break

        elif key == ord("c"):
            cleared = recognizer.get_word()
            if cleared:
                print(f"  Cleared: {cleared}")

        elif key == ord(" "):
            word = recognizer.get_word()
            if not word:
                print("  (no letters yet — sign something first)")
                continue

            print(f"\n  You signed: {word}")
            history.append(("received", word))

            # Ask Claude
            print("  Thinking...")
            response = ask_claude(word, history)
            history.append(("sent", response))

            # Display response as ASL images
            if not display_response(display, response):
                break  # user pressed Q during display

            # Back to listening
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
    print("Done.")


if __name__ == "__main__":
    main()
