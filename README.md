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