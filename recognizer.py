#!/usr/bin/env python3
#
# Sign Language Chat — Recognizer Module
#
# YOLO-based ASL letter detection, letter accumulation, and border color detection.
#

import cv2
import numpy as np
from ultralytics import YOLO
import config


class LetterAccumulator:
    """Confirms a letter detection only after N consecutive frames agree."""

    def __init__(self, required_frames=config.ACCUMULATION_FRAMES,
                 max_gap=config.MAX_GAP_FRAMES):
        self.required_frames = required_frames
        self.max_gap = max_gap
        self.current_letter = None
        self.count = 0
        self.gap = 0
        self.confirmed_letter = None

    def update(self, detected_letter):
        """Feed a detection result (letter string or None).

        Returns the confirmed letter if threshold is reached, else None.
        """
        self.confirmed_letter = None

        if detected_letter is None:
            self.gap += 1
            if self.gap > self.max_gap:
                self.current_letter = None
                self.count = 0
            return None

        self.gap = 0

        if detected_letter == self.current_letter:
            self.count += 1
        else:
            self.current_letter = detected_letter
            self.count = 1

        if self.count >= self.required_frames:
            self.confirmed_letter = self.current_letter
            # Reset so we don't keep re-confirming
            self.count = 0
            self.current_letter = None
            return self.confirmed_letter

        return None

    def reset(self):
        """Reset accumulator state."""
        self.current_letter = None
        self.count = 0
        self.gap = 0
        self.confirmed_letter = None


class ASLRecognizer:
    """Detects ASL letters from camera frames using YOLO and accumulates them."""

    def __init__(self, model_path=config.MODEL_PATH):
        print(f"ASLRecognizer: loading model from {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"ASLRecognizer: classes = {self.class_names}")

        self.accumulator = LetterAccumulator()
        self.word_buffer = []  # Accumulated confirmed letters

    def detect_frame(self, frame):
        """Run YOLO detection on a single frame.

        Args:
            frame: BGR image from camera

        Returns:
            (best_letter, confidence, annotated_frame)
            best_letter is None if nothing detected above threshold
        """
        # Optional: blur to reduce moire from screen
        if config.BLUR_KERNEL_SIZE > 0:
            k = config.BLUR_KERNEL_SIZE
            processed = cv2.GaussianBlur(frame, (k, k), 0)
        else:
            processed = frame

        results = self.model.predict(
            processed,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.NMS_THRESHOLD,
            verbose=False,
        )

        best_letter = None
        best_conf = 0.0
        annotated = frame.copy()

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                letter = self.class_names[cls_id]
                x1, y1, x2, y2 = map(int, box)

                # Draw detection box
                color = (0, 255, 0) if conf > 0.6 else (0, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{letter} {conf:.2f}"
                cv2.putText(
                    annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                )

                if conf > best_conf:
                    best_conf = conf
                    best_letter = letter.upper()

        return best_letter, best_conf, annotated

    def process_frame(self, frame):
        """Detect and accumulate. Returns (confirmed_letter, best_letter, confidence, annotated_frame).

        confirmed_letter is non-None only when accumulator confirms a letter.
        """
        best_letter, conf, annotated = self.detect_frame(frame)
        confirmed = self.accumulator.update(best_letter)

        if confirmed:
            self.word_buffer.append(confirmed)

        # Draw accumulator status
        status = f"Detecting: {self.accumulator.current_letter or '?'} ({self.accumulator.count}/{self.accumulator.required_frames})"
        cv2.putText(
            annotated, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2,
        )

        # Draw accumulated word
        word = "".join(self.word_buffer)
        cv2.putText(
            annotated, f"Word: {word}", (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        return confirmed, best_letter, conf, annotated

    def get_word(self):
        """Return the accumulated word and clear the buffer."""
        word = "".join(self.word_buffer)
        self.word_buffer.clear()
        self.accumulator.reset()
        return word

    def clear(self):
        """Clear the word buffer and reset accumulator."""
        self.word_buffer.clear()
        self.accumulator.reset()


def detect_border_color(frame):
    """Detect the dominant border color from the edges of a camera frame.

    Samples pixels from all four edges (border region) and checks against
    known signal colors using HSV ranges.

    Returns: "green", "red", "cyan", or None
    """
    h, f_h = frame.shape[:2], frame.shape[0]
    f_w = frame.shape[1]

    # Sample border pixels from edges (outer 15% of frame)
    margin = int(min(f_h, f_w) * 0.15)

    top = frame[0:margin, :]
    bottom = frame[f_h - margin : f_h, :]
    left = frame[:, 0:margin]
    right = frame[:, f_w - margin : f_w]

    # Combine all border pixels
    border_pixels = np.vstack([
        top.reshape(-1, 3),
        bottom.reshape(-1, 3),
        left.reshape(-1, 3),
        right.reshape(-1, 3),
    ])

    # Convert to HSV
    border_img = border_pixels.reshape(1, -1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(border_img, cv2.COLOR_BGR2HSV)
    hsv_flat = hsv.reshape(-1, 3)
    total = len(hsv_flat)

    # Check green
    mask_green = cv2.inRange(hsv, np.array(config.HSV_GREEN_LOWER), np.array(config.HSV_GREEN_UPPER))
    green_ratio = np.count_nonzero(mask_green) / total

    # Check red (wraps around hue=0)
    mask_red1 = cv2.inRange(hsv, np.array(config.HSV_RED_LOWER_1), np.array(config.HSV_RED_UPPER_1))
    mask_red2 = cv2.inRange(hsv, np.array(config.HSV_RED_LOWER_2), np.array(config.HSV_RED_UPPER_2))
    red_ratio = (np.count_nonzero(mask_red1) + np.count_nonzero(mask_red2)) / total

    # Check cyan
    mask_cyan = cv2.inRange(hsv, np.array(config.HSV_CYAN_LOWER), np.array(config.HSV_CYAN_UPPER))
    cyan_ratio = np.count_nonzero(mask_cyan) / total

    threshold = config.BORDER_COLOR_MIN_RATIO

    # Return the color with the highest ratio above threshold
    ratios = {"green": green_ratio, "red": red_ratio, "cyan": cyan_ratio}
    best = max(ratios, key=ratios.get)
    if ratios[best] >= threshold:
        return best

    return None
