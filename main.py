#!/usr/bin/env python3
#
# Sign Language Chat — Main Entry Point
#
# Two Gretchen robots communicate via ASL letters displayed on screens.
# One displays letters, the other reads them with its camera, then they swap.
#
# Usage:
#   python main.py --role speaker_first              # This laptop speaks first
#   python main.py --role listener_first             # This laptop listens first
#   python main.py --role speaker_first --no-robot   # Without robot hardware
#

import argparse
import sys
import time
import math
import cv2

import config
from display import ASLDisplay
from recognizer import ASLRecognizer, detect_border_color
from protocol import TurnProtocol, State
from conversation import ConversationManager, DEMO_SCRIPT_A, DEMO_SCRIPT_B


def open_camera(use_robot):
    """Open camera — either Gretchen's or a USB webcam."""
    if use_robot:
        from gretchen.robot import Robot
        robot = Robot(
            config.ROBOT_MOTOR_DEV,
            config.ROBOT_CAMERA_DEV,
            angle_limit=math.radians(config.ROBOT_ANGLE_LIMIT_DEG),
        )
        robot.start()
        robot.center()
        return robot, robot.camera
    else:
        cap = cv2.VideoCapture(config.CAMERA_DEV)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
        if not cap.isOpened():
            print("Error: cannot open camera")
            sys.exit(1)
        return None, cap


def read_frame(camera, use_robot):
    """Read a frame from the camera."""
    if use_robot:
        ret, frame, _ = camera.getImage()
        return ret, frame
    else:
        return camera.read()


def log_send(word):
    """Print a clear 'sending' line to the terminal."""
    letters = " ".join(word)
    print(f"  >>> SENDING: {word}  [{letters}]")


def log_receive(word):
    """Print a clear 'received' line to the terminal."""
    print(f"  <<< RECEIVED: {word}")


def log_letter(letter, index, total):
    """Print letter progress inline."""
    print(f"    [{index+1}/{total}] {letter}", end="  ", flush=True)


def speak_word(word, display, protocol, camera, use_robot):
    """Display a word letter by letter with green border.

    Returns True if completed, False if interrupted by keypress.
    """
    log_send(word)

    for i, letter in enumerate(word):
        if letter not in config.LETTERS:
            print(f"    (skipping '{letter}')", end="  ", flush=True)
            continue

        # Show the letter with green border
        display.show_letter(letter, config.COLOR_GREEN)
        log_letter(letter, i, len(word))

        # Hold the letter for the configured duration
        start = time.time()
        while time.time() - start < config.LETTER_DISPLAY_TIME:
            key = cv2.waitKey(50)
            if key == 27:  # ESC
                print()
                return False

        # Brief pause between letters
        display.show_blank(config.COLOR_GREEN)
        start = time.time()
        while time.time() - start < config.LETTER_PAUSE_TIME:
            key = cv2.waitKey(50)
            if key == 27:
                print()
                return False

    print()  # newline after letter progress

    # Show the completed word briefly
    display.show_word(word, config.COLOR_GREEN)
    time.sleep(0.5)

    return True


def listen_for_word(recognizer, display, protocol, camera, use_robot):
    """Listen via camera until the other side signals done (red border).

    Returns the detected word, or None if interrupted.
    """
    print("  Listening for letters...")
    display.show_blank(config.COLOR_CYAN)
    recognizer.clear()

    while True:
        ok, frame = read_frame(camera, use_robot)
        if not ok:
            continue

        # Check for border color signal
        border = detect_border_color(frame)

        if border == "red":
            # Other side is done — collect the word
            word = recognizer.get_word()
            if word:
                log_receive(word)
            return word if word else None

        # Process frame for letter detection
        confirmed, best, conf, annotated = recognizer.process_frame(frame)
        if confirmed:
            so_far = "".join(recognizer.word_buffer)
            print(f"    + {confirmed}  (word so far: {so_far})")

        # Show camera feed
        cv2.imshow(config.CAMERA_WINDOW, annotated)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            return None


def main():
    parser = argparse.ArgumentParser(description="Gretchen Sign Language Chat")
    parser.add_argument(
        "--role",
        required=True,
        choices=["speaker_first", "listener_first"],
        help="This laptop's starting role",
    )
    parser.add_argument(
        "--no-robot",
        action="store_true",
        help="Run without robot hardware (use laptop webcam)",
    )
    parser.add_argument(
        "--script",
        action="store_true",
        help="Use demo script instead of response-based conversation",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Display in fullscreen mode",
    )
    args = parser.parse_args()

    use_robot = not args.no_robot
    is_speaker_first = args.role == "speaker_first"

    # --- Initialize ---
    print("Initializing...")
    display = ASLDisplay(fullscreen=args.fullscreen)
    recognizer = ASLRecognizer()
    protocol = TurnProtocol(starts_as_speaker=is_speaker_first)

    robot, camera = open_camera(use_robot)

    cv2.namedWindow(config.CAMERA_WINDOW)

    # Set up conversation
    if args.script:
        script = DEMO_SCRIPT_A if is_speaker_first else DEMO_SCRIPT_B
        conversation = ConversationManager(script=script)
    else:
        conversation = ConversationManager()

    print(f"Role: {'speaker first' if is_speaker_first else 'listener first'}")
    print("Press ESC to quit.\n")

    # Show initial state
    display.show_blank(protocol.get_border_color())

    # --- Main conversation loop ---
    last_received = None

    try:
        while True:
            if protocol.is_speaking:
                # Get next word to send
                word = conversation.get_next_word(last_received)
                if word is None:
                    print("Conversation complete!")
                    break

                print(f"\n--- SPEAKING: {word} ---")

                # Spell out the word
                if not speak_word(word, display, protocol, camera, use_robot):
                    break  # ESC pressed

                # Signal done
                protocol.finish_speaking()
                display.show_blank(config.COLOR_RED)
                print("  Signaling DONE (red border)")

                # Wait for done timeout, then switch to listening
                while protocol.is_done_speaking:
                    ok, frame = read_frame(camera, use_robot)
                    if ok:
                        cv2.imshow(config.CAMERA_WINDOW, frame)
                    event = protocol.update()
                    if event == "done_timeout":
                        print("  Switching to LISTENING")
                    key = cv2.waitKey(50)
                    if key == 27:
                        break

            elif protocol.is_listening:
                print(f"\n--- LISTENING ---")
                display.show_blank(config.COLOR_CYAN)

                # Wait for the other side to start showing letters (green border)
                print("  Waiting for other side to start speaking...")
                while True:
                    ok, frame = read_frame(camera, use_robot)
                    if not ok:
                        continue

                    border = detect_border_color(frame)
                    cv2.imshow(config.CAMERA_WINDOW, frame)

                    if border == "green":
                        print("  Detected GREEN border — other side is speaking")
                        break
                    if border == "red":
                        # Other side already done (missed the green)
                        protocol.state = State.SPEAKING
                        print("  Detected RED border — taking our turn")
                        break

                    key = cv2.waitKey(50)
                    if key == 27:
                        raise KeyboardInterrupt

                if protocol.is_speaking:
                    continue  # Got turn signal, loop back to speaking

                # Now listen for letters until red border
                word = listen_for_word(recognizer, display, protocol, camera, use_robot)
                if word is None:
                    # Check if ESC or just empty
                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                    # Empty word, keep listening
                    continue

                conversation.receive_word(word)
                last_received = word

                # Show what we received
                display.show_word(f"GOT: {word}", config.COLOR_CYAN)
                time.sleep(1.5)

                # Now it's our turn to speak
                protocol.start_speaking()

            else:
                # Idle / waiting
                key = cv2.waitKey(100)
                if key == 27:
                    break
                ok, frame = read_frame(camera, use_robot)
                if ok:
                    border = detect_border_color(frame)
                    protocol.update(border)
                    cv2.imshow(config.CAMERA_WINDOW, frame)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    # --- Cleanup ---
    print("\n" + "=" * 40)
    print("  CONVERSATION LOG")
    print("=" * 40)
    for direction, word in conversation.get_history():
        if direction == "sent":
            print(f"  >>> SENT:     {word}")
        else:
            print(f"  <<< RECEIVED: {word}")
    print("=" * 40)

    display.close()
    cv2.destroyAllWindows()
    if use_robot and robot:
        robot.center()

    print("Done.")


if __name__ == "__main__":
    main()
