#!/usr/bin/env python3
#
# Sign Language Chat — Display Module
#
# Shows ASL letter images fullscreen with colored borders for signaling.
#

import cv2
import numpy as np
import os
import config


class ASLDisplay:
    """Displays ASL letter images in a window with colored borders for signaling."""

    def __init__(self, fullscreen=False):
        self.images = {}
        self._load_images()
        self.border_color = config.COLOR_GRAY

        cv2.namedWindow(config.DISPLAY_WINDOW, cv2.WINDOW_NORMAL)
        if fullscreen:
            cv2.setWindowProperty(
                config.DISPLAY_WINDOW,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN,
            )
        else:
            cv2.resizeWindow(config.DISPLAY_WINDOW, config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT)

    def _load_images(self):
        """Load one image per letter from the images directory."""
        for letter in config.LETTERS:
            path = os.path.join(config.IMAGES_DIR, f"{letter}.jpg")
            if os.path.exists(path):
                img = cv2.imread(path)
                if img is not None:
                    self.images[letter] = img
            else:
                # Try png
                path_png = os.path.join(config.IMAGES_DIR, f"{letter}.png")
                if os.path.exists(path_png):
                    img = cv2.imread(path_png)
                    if img is not None:
                        self.images[letter] = img

        loaded = list(self.images.keys())
        print(f"ASLDisplay: loaded {len(loaded)} letter images: {' '.join(loaded)}")

    def _get_screen_size(self):
        """Get the display window size."""
        return config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT

    def show_letter(self, letter, border_color=None):
        """Display a letter image with a colored border.

        Args:
            letter: Single uppercase letter (A-Y, excluding J and Z)
            border_color: BGR tuple for the border, or None to keep current
        """
        if border_color is not None:
            self.border_color = border_color

        if letter not in self.images:
            print(f"ASLDisplay: no image for letter '{letter}'")
            self.show_blank(self.border_color)
            return

        screen_w, screen_h = self._get_screen_size()
        bw = config.BORDER_WIDTH

        # Create canvas filled with border color
        canvas = np.full((screen_h, screen_w, 3), self.border_color, dtype=np.uint8)

        # Resize letter image to fit inside the border
        inner_w = screen_w - 2 * bw
        inner_h = screen_h - 2 * bw

        img = self.images[letter]
        h, w = img.shape[:2]

        # Maintain aspect ratio
        scale = min(inner_w / w, inner_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center the image within the inner area
        x_off = bw + (inner_w - new_w) // 2
        y_off = bw + (inner_h - new_h) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        # Add letter label in top-left corner
        cv2.putText(
            canvas,
            letter,
            (bw + 10, bw + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            (255, 255, 255),
            3,
        )

        cv2.imshow(config.DISPLAY_WINDOW, canvas)

    def show_blank(self, border_color=None):
        """Show a blank screen with just the border color."""
        if border_color is not None:
            self.border_color = border_color

        screen_w, screen_h = self._get_screen_size()
        canvas = np.full((screen_h, screen_w, 3), self.border_color, dtype=np.uint8)

        # Black center
        bw = config.BORDER_WIDTH
        canvas[bw : screen_h - bw, bw : screen_w - bw] = (0, 0, 0)

        cv2.imshow(config.DISPLAY_WINDOW, canvas)

    def show_word(self, word, border_color=None):
        """Display a word as large text on screen with border."""
        if border_color is not None:
            self.border_color = border_color

        screen_w, screen_h = self._get_screen_size()
        bw = config.BORDER_WIDTH
        canvas = np.full((screen_h, screen_w, 3), self.border_color, dtype=np.uint8)
        canvas[bw : screen_h - bw, bw : screen_w - bw] = (0, 0, 0)

        # Scale text to fit
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 4
        font_scale = 3.0

        text_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
        while text_size[0] > screen_w - 4 * bw and font_scale > 0.5:
            font_scale -= 0.5
            text_size = cv2.getTextSize(word, font, font_scale, thickness)[0]

        x = (screen_w - text_size[0]) // 2
        y = (screen_h + text_size[1]) // 2
        cv2.putText(canvas, word, (x, y), font, font_scale, (255, 255, 255), thickness)

        cv2.imshow(config.DISPLAY_WINDOW, canvas)

    def close(self):
        """Close the display window."""
        cv2.destroyWindow(config.DISPLAY_WINDOW)
