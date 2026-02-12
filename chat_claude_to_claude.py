#!/usr/bin/env python3
"""
Claude-to-Claude Sign Language Chat.

Two laptops, each running this script, have a real conversation via ASL.
Claude decides what to say on each side — no scripted responses.

Laptop A displays ASL letters → Laptop B's camera reads them →
Claude on B thinks of a response → B displays it → A reads it → repeat.

Synchronization uses colored screen borders:
    GREEN  = I am showing a letter, read it
    RED    = I am done, your turn
    CYAN   = I am listening / ready

Usage:
    # Laptop A (speaks first):
    python chat_claude_to_claude.py --role speaker --camera 0

    # Laptop B (listens first):
    python chat_claude_to_claude.py --role listener --camera 0

Keys:
    Q/ESC — quit
"""

import sys
import os
import argparse
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import config
from recognizer_mediapipe import MediaPipeRecognizer
from recognizer import detect_border_color
from display import ASLDisplay

DETECT_INTERVAL = 1.0 / 5.0

# Use own images for display
OWN_IMAGES_DIR = os.path.join(config.BASE_DIR, "images", "own")


def ask_claude(received_word, history, is_opening):
    """Ask Claude for a response via the claude CLI."""
    history_lines = []
    for direction, w in history:
        speaker = "Them" if direction == "received" else "Me"
        history_lines.append(f"  {speaker}: {w}")
    history_text = "\n".join(history_lines) if history_lines else "  (none yet)"

    if is_opening:
        prompt = (
            "You are Gretchen, a small humanoid robot starting a sign language "
            "conversation with another Gretchen robot.\n\n"
            "Pick ONE short greeting word (1-8 letters) using ONLY these letters: "
            "A B C D E F G H I K L M N O P Q R S T U V W X Y\n"
            "(No J or Z — those need motion in ASL.)\n\n"
            "Just the word, nothing else. No punctuation, no explanation."
        )
    else:
        prompt = (
            f"You are Gretchen, a small humanoid robot chatting in sign language "
            f"with another Gretchen robot.\n\n"
            f"Conversation so far:\n{history_text}\n  Them: {received_word}\n\n"
            f"Reply with ONE short word (1-8 letters) using ONLY these letters: "
            f"A B C D E F G H I K L M N O P Q R S T U V W X Y\n"
            f"(No J or Z — those need motion in ASL.)\n\n"
            f"Be creative and conversational. Don't just echo back the same word.\n"
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
        print("  (claude CLI not found)")
        return "HI"
    except Exception as e:
        print(f"  (Claude error: {e})")
        return "HI"


def speak_word(word, display):
    """Display a word letter by letter with green border.

    Returns True if completed, False if user pressed Q/ESC.
    """
    letters = " ".join(word)
    print(f"  >>> SENDING: {word}  [{letters}]")

    for i, letter in enumerate(word):
        if letter not in config.LETTERS:
            continue

        display.show_letter(letter, config.COLOR_GREEN)
        print(f"    [{i+1}/{len(word)}] {letter}", end="  ", flush=True)

        start = time.time()
        while time.time() - start < config.LETTER_DISPLAY_TIME:
            key = cv2.waitKey(50)
            if key == 27 or key == ord("q"):
                print()
                return False

        # Pause between letters
        display.show_blank(config.COLOR_GREEN)
        start = time.time()
        while time.time() - start < config.LETTER_PAUSE_TIME:
            cv2.waitKey(50)

    print()

    # Show full word briefly
    display.show_word(word, config.COLOR_GREEN)
    start = time.time()
    while time.time() - start < 1.0:
        cv2.waitKey(50)

    return True


def signal_done(display):
    """Show red border for 2 seconds to signal we're done talking."""
    print("  Signaling DONE (red border)")
    display.show_blank(config.COLOR_RED)
    start = time.time()
    while time.time() - start < 2.0:
        cv2.waitKey(50)


def wait_for_green(cap, display):
    """Show cyan border and wait until the other side shows green.

    Returns True when green detected, False if user quit.
    """
    print("  Waiting for other side to start speaking...")
    display.show_blank(config.COLOR_CYAN)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        border = detect_border_color(frame)
        cv2.imshow("ASL Camera", frame)

        if border == "green":
            print("  Detected GREEN — other side is speaking")
            return True
        if border == "red":
            # They finished before we even noticed green — treat as our turn
            print("  Detected RED — missed their message, taking turn")
            return True

        key = cv2.waitKey(50)
        if key == 27 or key == ord("q"):
            return False


def listen_for_word(cap, recognizer, display):
    """Read letters from camera until the other side signals done (red border).

    Returns the detected word, or None if interrupted.
    """
    print("  Listening for letters...")
    display.show_blank(config.COLOR_CYAN)
    recognizer.clear()
    last_detect_time = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Check border color
        border = detect_border_color(frame)

        if border == "red":
            word = recognizer.get_word()
            if word:
                print(f"  <<< RECEIVED: {word}")
            return word if word else None

        # Detect letters
        now = time.time()
        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now
            confirmed, _, _, annotated = recognizer.process_frame(frame)
            if confirmed:
                so_far = "".join(recognizer.word_buffer)
                print(f"    + {confirmed}  (word so far: {so_far})")
        else:
            annotated = frame

        cv2.imshow("ASL Camera", annotated)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            return None


def main():
    parser = argparse.ArgumentParser(description="Claude-to-Claude ASL Chat")
    parser.add_argument(
        "--role", required=True, choices=["speaker", "listener"],
        help="This laptop's starting role",
    )
    parser.add_argument(
        "--camera", default=config.CAMERA_DEV,
        help="Camera device path or index (default: /dev/grt_cam)",
    )
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument(
        "--rounds", type=int, default=10,
        help="Max conversation rounds before stopping (default: 10)",
    )
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

    print()
    print("=== Claude-to-Claude ASL Chat ===")
    print(f"Role: {args.role}")
    print(f"Max rounds: {args.rounds}")
    print("Q/ESC = quit")
    print()

    history = []
    speaking = args.role == "speaker"
    rounds = 0

    try:
        while rounds < args.rounds:
            if speaking:
                # --- OUR TURN TO SPEAK ---
                rounds += 1
                print(f"\n--- ROUND {rounds}: SPEAKING ---")

                # Ask Claude what to say
                if not history:
                    print("  Asking Claude for an opening word...")
                    word = ask_claude(None, history, is_opening=True)
                else:
                    last_received = history[-1][1] if history and history[-1][0] == "received" else None
                    print("  Asking Claude for a response...")
                    word = ask_claude(last_received, history, is_opening=False)

                print(f"  Claude chose: {word}")
                history.append(("sent", word))

                # Display the word as ASL letters
                if not speak_word(word, display):
                    break

                # Signal done
                signal_done(display)

                # Switch to listening
                speaking = False

            else:
                # --- OUR TURN TO LISTEN ---
                print(f"\n--- LISTENING ---")

                # Wait for the other side to start (green border)
                if not wait_for_green(cap, display):
                    break

                # Read letters until red border
                word = listen_for_word(cap, recognizer, display)
                if word is None:
                    # Could be ESC or just empty message
                    print("  (no word detected, retrying...)")
                    continue

                history.append(("received", word))

                # Show what we got
                display.show_word(f"GOT: {word}", config.COLOR_CYAN)
                start = time.time()
                while time.time() - start < 1.5:
                    cv2.waitKey(50)

                # Switch to speaking
                speaking = True

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # --- Summary ---
    print("\n" + "=" * 40)
    print("  CONVERSATION LOG")
    print("=" * 40)
    for direction, w in history:
        arrow = ">>>" if direction == "sent" else "<<<"
        label = "SENT" if direction == "sent" else "RECEIVED"
        print(f"  {arrow} {label:10s} {w}")
    print("=" * 40)

    cap.release()
    display.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()
