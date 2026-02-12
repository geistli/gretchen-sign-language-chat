#!/usr/bin/env python3
#
# Sign Language Chat — MediaPipe Recognizer
#
# Alternative to recognizer.py using MediaPipe's GestureRecognizer
# instead of YOLO. Uses hand landmark detection which may handle
# screen-to-camera scenarios differently.
#

import os
import cv2
import numpy as np
import mediapipe as mp
import config

MPImage = mp.Image
MPImageFormat = mp.ImageFormat
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

MEDIAPIPE_MODEL_PATH = os.path.join(config.BASE_DIR, "model", "asl_finger_spelling.task")


class LetterAccumulator:
    """Confirms a letter detection only after N consecutive frames agree."""

    def __init__(self, required_frames=config.ACCUMULATION_FRAMES,
                 max_gap=config.MAX_GAP_FRAMES):
        self.required_frames = required_frames
        self.max_gap = max_gap
        self.current_letter = None
        self.count = 0
        self.gap = 0

    def update(self, detected_letter):
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
            confirmed = self.current_letter
            self.count = 0
            self.current_letter = None
            return confirmed

        return None

    def reset(self):
        self.current_letter = None
        self.count = 0
        self.gap = 0


class MediaPipeRecognizer:
    """Detects ASL letters using MediaPipe GestureRecognizer."""

    def __init__(self, model_path=MEDIAPIPE_MODEL_PATH):
        print(f"MediaPipeRecognizer: loading model from {model_path}")

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.recognizer = GestureRecognizer.create_from_options(options)

        self.accumulator = LetterAccumulator()
        self.word_buffer = []
        print("MediaPipeRecognizer: ready")

    def detect_frame(self, frame):
        """Run MediaPipe gesture recognition on a single frame.

        Returns:
            (best_letter, confidence, annotated_frame)
        """
        annotated = frame.copy()

        # Convert BGR to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = MPImage(image_format=MPImageFormat.SRGB, data=rgb)

        result = self.recognizer.recognize(mp_image)

        best_letter = None
        best_conf = 0.0

        # Draw hand landmarks
        if result.hand_landmarks:
            for hand_landmarks in result.hand_landmarks:
                h, w = annotated.shape[:2]
                # Draw connections between landmarks
                connections = [
                    (0,1),(1,2),(2,3),(3,4),       # thumb
                    (0,5),(5,6),(6,7),(7,8),       # index
                    (0,9),(9,10),(10,11),(11,12),   # middle
                    (0,13),(13,14),(14,15),(15,16), # ring
                    (0,17),(17,18),(18,19),(19,20), # pinky
                    (5,9),(9,13),(13,17),           # palm
                ]
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
                for i, j in connections:
                    cv2.line(annotated, points[i], points[j], (0, 255, 0), 2)
                for pt in points:
                    cv2.circle(annotated, pt, 4, (0, 0, 255), -1)

        # Get gesture classification
        if result.gestures:
            for gesture_list in result.gestures:
                if gesture_list:
                    gesture = gesture_list[0]
                    letter = gesture.category_name.upper()
                    conf = gesture.score

                    if len(letter) != 1 or letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                        continue

                    if conf > best_conf:
                        best_conf = conf
                        best_letter = letter

                    # Draw label
                    label = f"{letter} {conf:.2f}"
                    color = (0, 255, 0) if conf > 0.6 else (0, 255, 255)
                    cv2.putText(
                        annotated, label, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2,
                    )

        return best_letter, best_conf, annotated

    def process_frame(self, frame):
        """Detect and accumulate.

        Returns (confirmed_letter, best_letter, confidence, annotated_frame).
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

    def reset(self):
        self.accumulator.reset()
