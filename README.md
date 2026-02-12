# Gretchen Sign Language Chat

Two Gretchen robots communicate via ASL (American Sign Language) letters displayed on their laptop screens. One robot displays letters, the other reads them with its camera using YOLO object detection, then they swap roles.

## How It Works

- **Display**: One laptop shows ASL hand sign photos one letter at a time
- **Recognition**: The other laptop's camera detects the letters using YOLO or MediaPipe gesture recognition
- **Turn-taking**: Colored screen borders signal whose turn it is (green = showing letter, red = done, cyan = listening)
- **Same code on both laptops**, differentiated by `--role speaker_first` or `--role listener_first`

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/geistli/gretchen-sign-language-chat.git
cd gretchen-sign-language-chat
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

**With uv (recommended):**
```bash
uv pip install ultralytics torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
uv pip install mediapipe
```

**With pip:**
```bash
pip install ultralytics torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
pip install mediapipe
```

The `--extra-index-url` pulls CPU-only PyTorch builds (~200MB instead of ~2GB with CUDA). No GPU needed.

### 4. Download the recognition models

**YOLO model** (object detection approach):
```bash
python tools/download_model.py
```
Downloads the pretrained YOLOv8s ASL model (~22MB) from HuggingFace into `model/`.

**MediaPipe model** (hand landmark approach):
```bash
curl -L -o model/asl_finger_spelling.task \
  "https://github.com/yoshan0921/asl-practice-app/raw/master/client/public/model/asl_finger_spelling.task"
```
Downloads the MediaPipe gesture recognizer (~8MB). This model detects hand skeleton landmarks and classifies the gesture, which may work better for screen-to-camera detection.

### 5. Get ASL alphabet images

**Option A — Kaggle dataset (best detection accuracy):**

1. Create a Kaggle account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings → API → Create New Token
3. Save the downloaded `kaggle.json`:
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```
4. Install kaggle CLI and download:
   ```bash
   pip install kaggle    # or: uv pip install kaggle
   python tools/download_images.py --kaggle
   ```

**Option B — Placeholder images (for testing the pipeline):**
```bash
python tools/download_images.py --placeholders
```

These are simple letter-on-circle images. The YOLO model won't detect them as real hand signs, but they work for testing the display and protocol.

### 6. Camera setup

The default camera is `/dev/grt_cam` (Gretchen's camera). To change it, edit `CAMERA_DEV` in `config.py` or pass `--camera` to the test tools.

## Usage

### Test the display
```bash
python tools/test_display.py          # Cycle all 24 letters
python tools/test_display.py HELLO    # Show a specific word
```
Keys: Space = next, Backspace = prev, G/R/C = border color, A = auto mode, ESC = quit

### Test the recognizer (YOLO)
```bash
python tools/test_recognizer.py                # Default camera
python tools/test_recognizer.py --camera 0     # Webcam at index 0
```
Keys: C = clear word, B = toggle border detection, Q/ESC = quit

### Test the recognizer (MediaPipe)
```bash
python tools/test_recognizer_mediapipe.py                # Default camera
python tools/test_recognizer_mediapipe.py --camera 0     # Webcam at index 0
```
Keys: C = clear word, Q/ESC = quit

MediaPipe draws hand skeleton landmarks and may be faster on CPU. Try both to see which works better for your setup.

### Run the full chat

**Laptop A (speaks first):**
```bash
python main.py --role speaker_first --no-robot
```

**Laptop B (listens first):**
```bash
python main.py --role listener_first --no-robot
```

**With Gretchen robot hardware:**
```bash
python main.py --role speaker_first
```

**With fullscreen display:**
```bash
python main.py --role speaker_first --no-robot --fullscreen
```

## Project Structure

```
├── main.py              # Entry point, orchestrates conversation
├── config.py            # All constants (thresholds, colors, paths)
├── display.py           # Show ASL images in window with signal border
├── recognizer.py            # YOLO detection + letter accumulation + border color detection
├── recognizer_mediapipe.py  # MediaPipe alternative recognizer
├── protocol.py              # Turn-taking state machine
├── conversation.py          # Word/message management and responses
├── model/
│   ├── (yolov8s_asl.pt)            # YOLO weights (not in git)
│   └── (asl_finger_spelling.task)  # MediaPipe model (not in git)
├── images/
│   └── (A.jpg ... Y.jpg)   # One photo per letter (not in git)
└── tools/
    ├── download_model.py            # Download YOLO model from HuggingFace
    ├── download_images.py           # Download/generate ASL images
    ├── test_display.py              # Standalone: cycle through letters
    ├── test_recognizer.py           # Test YOLO recognizer from camera
    └── test_recognizer_mediapipe.py # Test MediaPipe recognizer from camera
```

## ASL Alphabet

24 static letters are supported: A B C D E F G H I K L M N O P Q R S T U V W X Y

J and Z are excluded because they require motion.

## Troubleshooting

- **Model won't load**: Make sure you ran `python tools/download_model.py` and `model/yolov8s_asl.pt` exists
- **No images**: Run `python tools/download_images.py --check` to see which are missing
- **Wrong camera**: Change `CAMERA_DEV` in `config.py` or use `--camera 0` / `--camera 1`
- **Poor detection from screen**: The YOLO model is trained on real hand photos. Filming a screen can cause glare/moire. Try adjusting `BLUR_KERNEL_SIZE` and `CONFIDENCE_THRESHOLD` in `config.py`
