import os
import sys
import argparse

# must be set before cv2 import on Windows
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GLOG_logtostderr", "0")

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import urllib.request
import time
import numpy as np
import pyglet
import threading
import keras
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)


parser = argparse.ArgumentParser(description="Gesture-controlled selfie camera")
parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
parser.add_argument("--time", type=float, default=3.0, help="Countdown duration in seconds")
parser.add_argument("--path", type=str, default="selfie.jpg", help="File path to save captured image")
args = parser.parse_args()


UPDATE_HZ = 30
HAND_MODEL = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
GESTURE_MODEL = os.path.join(os.path.dirname(__file__), 'gesture_recognition.keras')
LABEL_NAMES = ['dislike', 'like', 'peace', 'rock', 'stop']
GESTURE_DISPLAY = {'like': 'Thumbs up', 'dislike': 'Thumbs down', 'stop': 'Stop', 'peace': 'Peace', 'rock': 'Rock'}
IMG_SIZE = 64
HAND_PAD = 0.15
CONF_THRESH = 0.6  # minimum confidence to consider a gesture valid
GESTURE_STREAK = 4  # number of consecutive frames with the same gesture before it is considered valid
LIKE_STREAK = 10  # number of consecutive frames with the the "like" gesture before it is triggered, to reduce false positives
CAPTURE_DISPLAY = 2.5  # seconds to display "Saved!" message after capture
EXPOSURE_ALPHA = 0.92  # smoothing factor for long exposure effect (0-1, closer to 1 is smoother)
CHROMA_DIV = 50  # divisor for chromatic shift amount (higher is less shift)
DEBUG_HAND = False

# UI colors in BGR
COL_BG_PANEL = (30, 28, 28)
COL_ACCENT = (255, 136, 0)    # Apple Blue
COL_TEXT = (255, 255, 255)
COL_TEXT_DARK = (30, 28, 28)
COL_OK = (89, 199, 52)        # Apple Green
COL_WARN = (40, 141, 255)     # Apple Orange


# --- camera ---

class CameraThread:
    def __init__(self, video_id):
        backend = cv2.CAP_MSMF if sys.platform == 'win32' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(video_id, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._set_max_resolution()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _set_max_resolution(self):
        for w, h in [(1920, 1080), (1280, 720), (640, 480)]:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (w, h):
                print(f"Camera resolution: {w}x{h}")
                return

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


STATE_IDLE = 'idle'
STATE_COUNTDOWN = 'countdown'
STATE_CAPTURED = 'captured'


# --- drawing ---

def px(v):
    return max(1, int(round(v * SCALE)))


def draw_text_centered(frame, text, center, font_scale, color, thickness, outline=True):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness)
    x = int(center[0] - w / 2)
    y = int(center[1] + h / 2)
    if outline:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, COL_TEXT_DARK, thickness + px(3), cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def draw_panel(frame, top_left, bottom_right, color=COL_BG_PANEL, alpha=0.75):
    np.copyto(_overlay_buf, frame)
    cv2.rectangle(_overlay_buf, top_left, bottom_right, color, -1, cv2.LINE_AA)
    cv2.addWeighted(_overlay_buf, alpha, frame, 1 - alpha, 0, frame)


def draw_hand_overlay(frame, hands, gesture, conf):
    for i, (_, bbox) in enumerate(hands):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), COL_ACCENT, px(2), cv2.LINE_AA)
        if i == 0 and gesture:
            label = f"{GESTURE_DISPLAY.get(gesture, gesture)}  {conf:.0%}"
            label_y = max(y1 - px(10), px(20))
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, SCALE * 0.7,
                        COL_TEXT_DARK, px(4), cv2.LINE_AA)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, SCALE * 0.7,
                        COL_TEXT, px(2), cv2.LINE_AA)


def draw_hud(frame, app_state, chroma, exposure):
    hints = {
        STATE_IDLE: 'Thumbs up -> start  |  Rock -> toggle chroma  |  Peace -> toggle long exposure',
        STATE_COUNTDOWN: 'Thumbs down -> cancel',
    }
    hint = hints.get(app_state)
    if not hint:
        return
    bar_h = px(50)
    draw_panel(frame, (0, WINDOW_HEIGHT - bar_h), (WINDOW_WIDTH, WINDOW_HEIGHT))
    draw_text_centered(frame, hint,
                       (WINDOW_WIDTH // 2, WINDOW_HEIGHT - bar_h // 2),
                       SCALE * 0.45, COL_TEXT, px(1), outline=False)
    fs, th = SCALE * 0.45, px(1)
    cy = WINDOW_HEIGHT - bar_h // 2
    for i, (label, active) in enumerate([('Chroma', chroma), ('Long Exposure', exposure)]):
        color = COL_ACCENT if active else (100, 100, 100)
        tw = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, th)[0][0]
        x = WINDOW_WIDTH - px(16) - tw - i * (tw + px(24))
        _, th2 = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, fs, th)[0]
        cv2.putText(frame, label, (x, cy + th2 // 2),
                    cv2.FONT_HERSHEY_DUPLEX, fs, color, th, cv2.LINE_AA)



# --- detection ---

def detect_hands(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ts = int(time.monotonic() * 1000)
    result = _landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    out = []
    for lms in result.hand_landmarks:
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms])
        mn, mx = pts.min(0).astype(int), pts.max(0).astype(int)
        pad = ((mx - mn) * HAND_PAD).astype(int)
        x1, y1 = np.maximum([0, 0], mn - pad)
        x2, y2 = np.minimum([w, h], mx + pad)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size:
            out.append((crop, (x1, y1, x2, y2)))
    return out


def classify_gesture(crop_bgr):
    if _model is None:
        return None, None
    img = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE)).astype('float32') / 255.
    probs = _model.predict(img.reshape(1, IMG_SIZE, IMG_SIZE, 3), verbose=0)[0]
    idx = int(probs.argmax())
    conf = float(probs[idx])
    if conf >= CONF_THRESH:
        return LABEL_NAMES[idx], conf
    return None, None


def _infer_worker(crop):
    result = classify_gesture(crop)
    with _infer_lock:
        _infer['cache'] = result
        _infer['busy'] = False


# --- filters ---

def apply_chromatic(frame):
    shift = max(1, WINDOW_WIDTH // CHROMA_DIV)  # pixel shift amount based on resolutions
    b, g, r = cv2.split(frame)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    return cv2.merge([b, g, r])


def apply_exposure(frame):
    if _exposure['buf'] is None:
        _exposure['buf'] = frame.astype(np.float32)
    _exposure['buf'] = _exposure['buf'] * EXPOSURE_ALPHA + frame.astype(np.float32) * (1 - EXPOSURE_ALPHA)
    return np.clip(_exposure['buf'], 0, 255).astype(np.uint8)  # convert back to uint8


app = {
    'state': STATE_IDLE,
    'countdown': args.time,
    'capture_timer': 0.0,
    'streak_label': None,
    'streak_count': 0,
    'chroma': False,
    'exposure': False,
    'save_pending': False,
    'toast': '',
    'toast_timer': 0.0,
}


# --- update helpers ---

def _poll_gesture(hands):
    if hands:
        with _infer_lock:
            if not _infer["busy"]:
                _infer["busy"] = True
                threading.Thread(target=_infer_worker, args=(hands[0][0].copy(),), daemon=True).start()
        with _infer_lock:
            raw_label, raw_conf = _infer["cache"]
    else:
        with _infer_lock:
            _infer["cache"] = (None, None)
        raw_label, raw_conf = None, None

    if raw_label == app["streak_label"]:
        app["streak_count"] += 1
    else:
        app["streak_label"] = raw_label
        app["streak_count"] = 1
    gesture = raw_label if app["streak_count"] >= GESTURE_STREAK else None
    threshold = LIKE_STREAK if raw_label == "like" else GESTURE_STREAK
    gesture_triggered = raw_label if app["streak_count"] == threshold else None
    return gesture, gesture_triggered, raw_conf if gesture else None


def _tick_state(gesture_triggered, dt):
    if app["state"] != STATE_COUNTDOWN:
        if gesture_triggered == "rock":
            app["chroma"] = not app["chroma"]
            app["toast"] = "Chroma on" if app["chroma"] else "Chroma off"
            app["toast_timer"] = 1.5
        elif gesture_triggered == "peace":
            app["exposure"] = not app["exposure"]
            app["toast"] = "Long exposure on" if app["exposure"] else "Long exposure off"
            app["toast_timer"] = 1.5
            if not app["exposure"]:
                _exposure["buf"] = None

    if app["state"] == STATE_IDLE:
        if gesture_triggered == "like":
            app["state"] = STATE_COUNTDOWN  # start countdown
            app["countdown"] = args.time
    elif app["state"] == STATE_COUNTDOWN:
        if gesture_triggered == "dislike":
            app["state"] = STATE_IDLE  # cancel countdown
            app["countdown"] = args.time
        else:
            app["countdown"] -= dt
            if app["countdown"] <= 0:
                app["state"] = STATE_CAPTURED  # capture photo
                app["countdown"] = args.time
                app["capture_timer"] = 2.5
                app["save_pending"] = True
    elif app["state"] == STATE_CAPTURED:
        app["capture_timer"] -= dt
        if app["capture_timer"] <= 0:
            app["state"] = STATE_IDLE
    if app["toast_timer"] > 0:
        app["toast_timer"] -= dt


def _apply_filters(display):
    if app["chroma"]:
        display = apply_chromatic(display)
    if app["exposure"]:
        display = apply_exposure(display)
    if app["save_pending"]:
        cv2.imwrite(args.path, display)
        print(f"Saved to {args.path}")
        app["save_pending"] = False
    return display


def _draw(display, hands, gesture, conf):
    if app["state"] == STATE_COUNTDOWN:
        draw_text_centered(display, str(int(app["countdown"]) + 1),
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
                           SCALE * 5.0, COL_TEXT, px(6))
    elif app["state"] == STATE_CAPTURED:
        draw_text_centered(display, "Saved!",
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
                           SCALE * 3.5, COL_TEXT, px(5))
    draw_hand_overlay(display, hands, gesture, conf)
    if app["toast_timer"] > 0:
        draw_text_centered(display, app["toast"],
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT - px(80)),
                           SCALE * 0.6, COL_TEXT, px(1))
    draw_hud(display, app["state"], app["chroma"], app["exposure"])


def update(dt):
    frame = cap.read()
    if frame is None:
        return

    display = cv2.flip(frame, 1)
    hands = detect_hands(display)
    gesture, gesture_triggered, conf = _poll_gesture(hands)

    if DEBUG_HAND:
        tile_h = 200
        tiles = ([cv2.resize(c, (int(c.shape[1] * tile_h / c.shape[0]), tile_h)) for c, _ in hands]
                 if hands else [np.zeros((tile_h, tile_h, 3), np.uint8)])
        cv2.imshow("hand crop", np.hstack(tiles))
        cv2.waitKey(1)

    _tick_state(gesture_triggered, dt)
    display = _apply_filters(display)
    _draw(display, hands, gesture, conf)

    img = pyglet.image.ImageData(WINDOW_WIDTH, WINDOW_HEIGHT, "BGR", display.tobytes(), pitch=-WINDOW_WIDTH * 3)
    _display_tex.blit_into(img, 0, 0, 0)

# --- startup ---

print("Starting camera...")
cap = CameraThread(args.camera)

print("Waiting for first frame...")
first_frame = None
while first_frame is None:
    first_frame = cap.read()
print("Ready.")

WINDOW_HEIGHT, WINDOW_WIDTH = first_frame.shape[:2]
SCALE = min(WINDOW_WIDTH, WINDOW_HEIGHT) / 720.0
_overlay_buf = np.empty_like(first_frame)

# Ensure hand landmarker model is downloaded
if not os.path.exists(HAND_MODEL):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)

_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

if os.path.exists(GESTURE_MODEL):
    print("Loading gesture model...")
    _model = keras.models.load_model(GESTURE_MODEL)
else:
    print(f"WARNING: gesture model not found at {GESTURE_MODEL}, classification disabled")
    _model = None

_infer = {'cache': (None, None), 'busy': False}
_infer_lock = threading.Lock()
_exposure = {'buf': None}

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="Selfie Camera")
_display_tex = pyglet.image.Texture.create(WINDOW_WIDTH, WINDOW_HEIGHT)
display_sprite = pyglet.sprite.Sprite(_display_tex)

pyglet.clock.schedule_interval(update, 1 / UPDATE_HZ)


# --- events ---

@window.event
def on_draw():
    window.clear()
    display_sprite.draw()


@window.event
def on_key_press(symbol, _modifiers):
    global DEBUG_HAND
    key = pyglet.window.key
    if symbol == key.Q:
        window.close()
    elif symbol == key.D:
        DEBUG_HAND = not DEBUG_HAND
        if not DEBUG_HAND:
            cv2.destroyWindow('hand crop')
    elif symbol == key.SPACE:
        _tick_state('like', 0)
    elif symbol == key.ESCAPE:
        _tick_state('dislike', 0)
    elif symbol == key.C:
        _tick_state('rock', 0)
    elif symbol == key.E:
        _tick_state('peace', 0)


@window.event
def on_close():
    cap.release()
    _landmarker.close()


pyglet.app.run()
