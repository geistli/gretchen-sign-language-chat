#!/usr/bin/env python3
"""
Claude-to-Claude Sign Language Chat.

Two laptops, each running this script, have a real conversation via ASL.
Claude decides what to say on each side — no scripted responses.
Supports full sentences — spaces are shown as longer pauses between words,
and detected by the listener when no hand is visible for several frames.

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
import config
from recognizer_mediapipe import MediaPipeRecognizer
from recognizer import detect_border_color
from display import ASLDisplay

DETECT_INTERVAL = 1.0 / 5.0

# Frames with no hand detected before inserting a space
SPACE_GAP_FRAMES = 8

# Seconds of silence (no new letters) before sentence is considered done
SENTENCE_DONE_TIMEOUT = 3.0

# Use own images for display
OWN_IMAGES_DIR = os.path.join(config.BASE_DIR, "images", "own")


def ask_claude(received_text, history, is_opening):
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
            "Pick a short greeting (1-3 words, max 8 letters per word) using "
            "ONLY these letters: A B C D E F G H I K L M N O P Q R S T U V W X Y\n"
            "(No J or Z — those need motion in ASL.)\n\n"
            "Just the words separated by spaces, nothing else. "
            "No punctuation, no explanation."
        )
    else:
        prompt = (
            f"You are Gretchen, a small humanoid robot chatting in sign language "
            f"with another Gretchen robot.\n\n"
            f"Conversation so far:\n{history_text}\n  Them: {received_text}\n\n"
            f"Reply with a short response (1-3 words, max 8 letters per word) "
            f"using ONLY these letters: A B C D E F G H I K L M N O P Q R S T U V W X Y\n"
            f"(No J or Z — those need motion in ASL.)\n\n"
            f"Be creative and conversational. Don't just echo back the same words. "
            f"NEVER repeat something you already said.\n\n"
            f"Just the words separated by spaces, nothing else. "
            f"No punctuation, no explanation."
        )

    try:
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=30,
        )
        response = result.stdout.strip().upper()
        # Keep only valid ASL letters and spaces
        filtered = "".join(c for c in response if c in config.LETTERS or c == " ")
        # Clean up multiple spaces
        filtered = " ".join(filtered.split())
        return filtered if filtered else "HI"
    except subprocess.TimeoutExpired:
        print("  (Claude took too long, defaulting to HI)")
        return "HI"
    except FileNotFoundError:
        print("  (claude CLI not found)")
        return "HI"
    except Exception as e:
        print(f"  (Claude error: {e})")
        return "HI"


def speak_text(text, display):
    """Display text letter by letter with green border. Spaces become pauses.

    Returns True if completed, False if user pressed Q/ESC.
    """
    print(f"  >>> SENDING: {text}")

    letter_count = sum(1 for c in text if c != " ")
    letter_idx = 0

    for char in text:
        if char == " ":
            # Longer pause between words
            display.show_blank(config.COLOR_GREEN)
            start = time.time()
            while time.time() - start < 1.0:
                key = cv2.waitKey(50)
                if key == 27 or key == ord("q"):
                    return False
            continue

        if char not in config.LETTERS:
            continue

        letter_idx += 1
        display.show_letter(char, config.COLOR_GREEN)
        print(f"    [{letter_idx}/{letter_count}] {char}", end="  ", flush=True)

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

    # Show full text briefly
    display.show_word(text, config.COLOR_GREEN)
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
            print("  Detected RED — missed their message, taking turn")
            return True

        key = cv2.waitKey(50)
        if key == 27 or key == ord("q"):
            return False


def listen_for_sentence(cap, recognizer, display):
    """Read letters from camera until the other side signals done (red border).
    Detects spaces when no hand is visible for several frames.

    Returns the detected sentence, or None if interrupted.
    """
    print("  Listening for letters...")
    display.show_blank(config.COLOR_CYAN)
    recognizer.clear()

    sentence = []
    no_hand_frames = 0
    space_inserted = True
    last_detect_time = 0
    last_letter_time = None  # when the last letter was confirmed

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # Check border color
        border = detect_border_color(frame)

        if border == "red":
            # Other side is done
            text = "".join(sentence).strip()
            if text:
                print(f"  <<< RECEIVED: {text}")
            return text if text else None

        # Check silence timeout — sentence done if no new letters for 3s
        now = time.time()
        if sentence and last_letter_time and (now - last_letter_time) >= SENTENCE_DONE_TIMEOUT:
            text = "".join(sentence).strip()
            if text:
                print(f"  <<< RECEIVED (timeout): {text}")
                return text

        # Detect letters
        if now - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = now
            confirmed, best, _, annotated = recognizer.process_frame(frame)

            # Track hand presence for space detection
            if best is not None:
                no_hand_frames = 0
                space_inserted = False
            else:
                no_hand_frames += 1

            # Insert space when hand gone long enough
            if no_hand_frames >= SPACE_GAP_FRAMES and not space_inserted and sentence and sentence[-1] != " ":
                sentence.append(" ")
                space_inserted = True
                print("    [SPACE]")

            if confirmed:
                last_letter_time = now
                # Skip duplicate consecutive letters
                if not sentence or sentence[-1] != confirmed:
                    sentence.append(confirmed)
                    text = "".join(sentence)
                    print(f"    + {confirmed}  (so far: {text})")
        else:
            annotated = frame

        # Draw current sentence on camera feed
        text = "".join(sentence)
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, h - 50), (w, h), (0, 0, 0), -1)
        cv2.putText(annotated, text, (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

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
                    print("  Asking Claude for an opening...")
                    text = ask_claude(None, history, is_opening=True)
                else:
                    last_received = history[-1][1] if history and history[-1][0] == "received" else None
                    print("  Asking Claude for a response...")
                    text = ask_claude(last_received, history, is_opening=False)

                print(f"  Claude chose: {text}")
                history.append(("sent", text))

                # Display the text as ASL letters
                if not speak_text(text, display):
                    break

                # Signal done
                signal_done(display)

                # Switch to listening
                speaking = False

            else:
                # --- OUR TURN TO LISTEN ---
                print(f"\n--- LISTENING ---")

                # Listen for letters, sentence ends after 3s silence
                text = listen_for_sentence(cap, recognizer, display)
                if text is None:
                    print("  (nothing detected, retrying...)")
                    continue

                history.append(("received", text))

                # Show what we got
                display.show_word(f"GOT: {text}", config.COLOR_CYAN)
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
