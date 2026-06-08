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

### Results

Confusion matrices on the HaGRID test split and on the recorded images:

![Test set confusion matrices](02-dataset/conf-matrix.png)

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

On first run, the [MediaPipe Hand Landmarker model](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) (~7.5 MB) is downloaded automatically and cached in `03-camera-app/`.

```bash
python 03-camera-app/camera_app.py --time 5 --path selfie.jpg
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--camera` | `0` | Camera device ID |
| `--time` | `3.0` | Countdown duration in seconds |
| `--path` | `selfie.jpg` | File path for the saved image |

| Key | Action |
|-----|--------|
| `D` | Toggle hand crop debug window |
| `Q` | Quit |