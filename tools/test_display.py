#!/usr/bin/env python3
#
# Test tool: cycle through ASL letter images on screen.
#
# Usage:
#   python tools/test_display.py          # Cycle all letters
#   python tools/test_display.py HELLO    # Show specific word
#
# Keys:
#   Space/Enter — next letter
#   Backspace   — previous letter
#   G/R/C       — toggle border color (green/red/cyan)
#   ESC         — quit
#

import sys
import os
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import config
from display import ASLDisplay


def main():
    display = ASLDisplay()

    if len(sys.argv) > 1:
        # Show a specific word
        word = sys.argv[1].upper()
        letters = [c for c in word if c in config.LETTERS]
    else:
        letters = config.LETTERS

    if not letters:
        print("No valid letters to display!")
        return

    print(f"Displaying {len(letters)} letters: {' '.join(letters)}")
    print("Space/Enter=next, Backspace=prev, G/R/C=border color, ESC=quit")

    idx = 0
    border_color = config.COLOR_GREEN

    while True:
        letter = letters[idx % len(letters)]
        display.show_letter(letter, border_color)

        key = cv2.waitKey(0)

        if key == 27:  # ESC
            break
        elif key in (32, 13, ord('n')):  # Space, Enter, N
            idx += 1
            if idx >= len(letters):
                print("All letters shown!")
                idx = 0
        elif key == 8 or key == ord('p'):  # Backspace, P
            idx = max(0, idx - 1)
        elif key == ord('g'):
            border_color = config.COLOR_GREEN
            print("Border: GREEN")
        elif key == ord('r'):
            border_color = config.COLOR_RED
            print("Border: RED")
        elif key == ord('c'):
            border_color = config.COLOR_CYAN
            print("Border: CYAN")
        elif key == ord('a'):
            # Auto mode: cycle through all letters with timing
            print("Auto mode...")
            for i, l in enumerate(letters):
                display.show_letter(l, config.COLOR_GREEN)
                print(f"  {l} ({i+1}/{len(letters)})")
                k = cv2.waitKey(int(config.LETTER_DISPLAY_TIME * 1000))
                if k == 27:
                    break
                display.show_blank(config.COLOR_GREEN)
                cv2.waitKey(int(config.LETTER_PAUSE_TIME * 1000))
            # Show red border to signal done
            display.show_blank(config.COLOR_RED)
            print("  DONE (red border)")

    display.close()
    print("Done.")


if __name__ == "__main__":
    main()
