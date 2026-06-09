[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/cMaQVOgt)


# Assignment 5: Hand Pose Detection with a CNN

## Setup
1. Clone the repo and navigate to it via `cd assignment-05-cnn-Alphazerfall`.
2. Set up a virtual environment by running `python -m venv .venv`.
3. Activate the virtual environment using `.venv\Scripts\activate` on Windows and `source .venv/bin/activate` on Linux/Mac.
4. Install the required dependencies via `pip install -r requirements.txt`.
5. Place the HaGRID dataset sample (provided via GRIPS) in a folder called `data/` at the repo root. It should contain gesture subfolders (`like/`, `dislike/`, ...) and an `_annotations/` folder.

## 1. Exploring Hyperparameters

## 2. Gathering a Dataset

### Image Capture

Five gesture categories (**like**, **dislike**, **stop**, **rock**, **peace**) were recorded across three locations. Images were captured with an **iPhone 13 Pro** (portrait, resized to 1080×1920 or 1440×1920) and a **Logitech C920 Pro** webcam (1920×1080 landscape). All images are stored in `02-dataset/data_felix/`.

### Results

Confusion matrices on the HaGRID test split and on the recorded images:

![Test set confusion matrices](02-dataset/conf-matrix.png)

| Model | Train Accuracy | Val Accuracy | Val Loss |
|-------|---------------|--------------|----------|
| Baseline CNN | 96.50 % | 94.80 % | 0.1832 |
| VGG-based CNN | 99.50 % | 89.20 % | 0.3088 |

The Baseline CNN is the sequential model provided in the course (3x Conv2D + MaxPooling, Dropout, 2x Dense). The baseline performs better on the validation set even though it is simpler. The VGG model scores higher on training but drops off on validation, suggesting it is overfitting. The Baseline CNN also has far fewer parameters (297K / 1.13 MB vs. 15.3M / 58.4 MB), making it faster and lighter. For these reasons it was chosen as the model for Exercise 3.

### Annotation tool (`02-dataset/annotate.py`)

Expects images organised in gesture subfolders (same layout as HaGRID):

```
02-dataset/data_felix/
    dislike/  img1.jpg  img2.jpg  ...
    like/     ...
    peace/    ...
    ...
```

The gesture label is inferred from the subfolder name, so the first box drawn is automatically labelled with that gesture. Additional boxes default to `no_gesture`.

**Annotate your own photos:**
```bash
python 02-dataset/annotate.py --images 02-dataset/data_felix/ --output 02-dataset/annot-felix.json
```

**View HaGRID annotations (read-only):**
```bash
# All gestures
python 02-dataset/annotate.py --images data/ --load data/_annotations/ --readonly

# Single gesture
python 02-dataset/annotate.py --images data/ --load data/_annotations/stop.json --readonly
```

**Controls:**

| Key | Action |
|-----|--------|
| `1`–`9`, `0` | Select gesture label (like, dislike, stop, rock, peace, fist, ok, one, three, two_up) |
| `N` | Select `no_gesture` |
| Drag | Draw bounding box around a hand |
| `Z` | Undo last box |
| `S` | Save & advance to next image (advance only in `--readonly`) |
| `Q` | Save & quit |

## 3. Gesture-controlled Camera App

The app uses the [MediaPipe Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) to locate the hand in the frame and extract a crop, which is then passed to the gesture classifier. Only one hand is tracked at a time. The model file (~7.5 MB) is downloaded automatically on first run and cached in `03-camera-app/`.

```bash
python 03-camera-app/camera_app.py --time 5 --path selfie.jpg
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `0` | Camera device ID |
| `--time` | `3.0` | Countdown duration in seconds |
| `--path` | `selfie.jpg` | File path for the saved image |

After capture the app pauses for 2.5 seconds showing "Saved!" before accepting new gestures. Gestures must be held for ~4 frames before they register.

| Key | Gesture equivalent | Action |
|-----|--------------------|--------|
| `Space` | Thumbs up | Start countdown |
| `Esc` | Thumbs down | Cancel countdown |
| `C` | Rock | Toggle chromatic aberration |
| `E` | Peace | Toggle long exposure |
| `D` | — | Toggle hand crop debug window |
| `Q` | — | Quit |

**Chromatic aberration** (Rock / `C`) — splits the colour channels horizontally, giving a glitchy RGB fringe effect.

**Long exposure** (Peace / `E`) — blends each frame into a float accumulator, ghosting moving objects while keeping static parts sharp. The effect fades over roughly one second.