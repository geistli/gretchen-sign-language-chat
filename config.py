#!/usr/bin/env python3
#
# Sign Language Chat — Configuration
#
# All constants: paths, thresholds, colors, timing
#

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "yolov8s_asl.pt")
IMAGES_DIR = os.path.join(BASE_DIR, "images")

# --- ASL Alphabet ---
# 24 static letters (J and Z require motion, excluded)
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")

# --- YOLO Detection ---
CONFIDENCE_THRESHOLD = 0.40
NMS_THRESHOLD = 0.4

# --- Letter Accumulation ---
# Number of consecutive frames with same letter to confirm detection
ACCUMULATION_FRAMES = 8
# Maximum gap (frames without detection) before resetting accumulator
MAX_GAP_FRAMES = 3

# --- Display ---
BORDER_WIDTH = 40  # pixels, thick border for reliable color detection
DISPLAY_WINDOW = "ASL Display"
CAMERA_WINDOW = "ASL Camera"
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 600

# Letter display duration (seconds) — how long each letter is shown
LETTER_DISPLAY_TIME = 2.0
# Pause between letters (seconds)
LETTER_PAUSE_TIME = 0.5

# --- Border Signal Colors (BGR for OpenCV) ---
COLOR_GREEN = (0, 255, 0)      # "I am displaying a letter — read it"
COLOR_RED = (0, 0, 255)        # "I am done, your turn"
COLOR_CYAN = (255, 255, 0)     # "I am ready and listening"
COLOR_GRAY = (128, 128, 128)   # Idle / startup

# --- HSV Ranges for Border Detection ---
# These ranges detect the border color from the camera feed
# HSV ranges: (H_low, S_low, V_low), (H_high, S_high, V_high)
HSV_GREEN_LOWER = (35, 100, 100)
HSV_GREEN_UPPER = (85, 255, 255)

HSV_RED_LOWER_1 = (0, 100, 100)
HSV_RED_UPPER_1 = (10, 255, 255)
HSV_RED_LOWER_2 = (170, 100, 100)
HSV_RED_UPPER_2 = (180, 255, 255)

HSV_CYAN_LOWER = (80, 100, 100)
HSV_CYAN_UPPER = (100, 255, 255)

# Minimum ratio of border pixels matching a color to count as detected
BORDER_COLOR_MIN_RATIO = 0.3

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# --- Robot ---
ROBOT_MOTOR_DEV = "/dev/grt_motor"
ROBOT_CAMERA_DEV = "/dev/grt_cam"
ROBOT_ANGLE_LIMIT_DEG = 20

# --- Preprocessing ---
# Gaussian blur kernel size for reducing moire when detecting screen content
BLUR_KERNEL_SIZE = 5
