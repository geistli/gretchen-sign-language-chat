# Gretchen Sign Language Chat

Two Gretchen robots communicate via ASL (American Sign Language) letters displayed on their laptop screens. One robot displays letters, the other reads them with its camera using YOLO object detection, then they swap roles.

## How It Works

- **Display**: One laptop shows ASL hand sign photos one letter at a time
- **Recognition**: The other laptop's camera detects the letters using a pretrained YOLOv8 model
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
```

**With pip:**
```bash
pip install ultralytics torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```

The `--extra-index-url` pulls CPU-only PyTorch builds (~200MB instead of ~2GB with CUDA). No GPU needed.

### 4. Download the ASL YOLO model

```bash
python tools/download_model.py
```

This downloads the pretrained YOLOv8s model (~22MB) from HuggingFace into `model/`.

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

### Test the recognizer
```bash
python tools/test_recognizer.py                # Default camera
python tools/test_recognizer.py --camera 0     # Webcam at index 0
```
Keys: C = clear word, B = toggle border detection, ESC = quit

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
├── recognizer.py        # YOLO detection + letter accumulation + border color detection
├── protocol.py          # Turn-taking state machine
├── conversation.py      # Word/message management and responses
├── model/
│   └── (yolov8s_asl.pt)    # Downloaded pretrained weights (not in git)
├── images/
│   └── (A.jpg ... Y.jpg)   # One photo per letter (not in git)
└── tools/
    ├── download_model.py    # Download model from HuggingFace
    ├── download_images.py   # Download/generate ASL images
    ├── test_display.py      # Standalone: cycle through letters
    └── test_recognizer.py   # Standalone: detect ASL from camera
```

## ASL Alphabet

24 static letters are supported: A B C D E F G H I K L M N O P Q R S T U V W X Y

J and Z are excluded because they require motion.

## Troubleshooting

- **Model won't load**: Make sure you ran `python tools/download_model.py` and `model/yolov8s_asl.pt` exists
- **No images**: Run `python tools/download_images.py --check` to see which are missing
- **Wrong camera**: Change `CAMERA_DEV` in `config.py` or use `--camera 0` / `--camera 1`
- **Poor detection from screen**: The YOLO model is trained on real hand photos. Filming a screen can cause glare/moire. Try adjusting `BLUR_KERNEL_SIZE` and `CONFIDENCE_THRESHOLD` in `config.py`
