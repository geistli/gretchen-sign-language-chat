#!/usr/bin/env python3
#
# Test tool: display ASL letters for a message at 3 letters/second.
#
# Displays "HELLO I AM GRETCHEN" by default, or a custom message via args.
# Spaces between words are shown as a brief blank screen.
#
# Usage:
#   python tools/test_display.py                    # Default message
#   python tools/test_display.py "HI THERE"         # Custom message
#
# Press Q or ESC to quit early.
#

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import config
from display import ASLDisplay

LETTERS_PER_SECOND = 3
LETTER_MS = int(1000 / LETTERS_PER_SECOND)
SPACE_MS = 500  # pause between words


def main():
    display = ASLDisplay()

    message = sys.argv[1].upper() if len(sys.argv) > 1 else "HELLO I AM GRETCHEN"

    words = message.split()
    print(f"Message: {message}")
    print(f"Speed: {LETTERS_PER_SECOND} letters/sec")
    print("Q/ESC to quit\n")

    for wi, word in enumerate(words):
        print(f"  [{word}]", end="  ", flush=True)
        for letter in word:
            if letter not in config.LETTERS:
                print(f"(skip '{letter}')", end=" ", flush=True)
                continue

            display.show_letter(letter, config.COLOR_GREEN)
            print(letter, end=" ", flush=True)

            key = cv2.waitKey(LETTER_MS)
            if key == 27 or key == ord('q'):
                print("\n\nStopped.")
                display.close()
                return

        print()  # newline after each word

        # Pause between words
        if wi < len(words) - 1:
            display.show_blank(config.COLOR_GREEN)
            key = cv2.waitKey(SPACE_MS)
            if key == 27 or key == ord('q'):
                print("\nStopped.")
                display.close()
                return

    # Done
    display.show_word(message, config.COLOR_RED)
    print(f"\nDone! Showing full message. Q/ESC to close.")
    while True:
        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):
            break

    display.close()


if __name__ == "__main__":
    main()
