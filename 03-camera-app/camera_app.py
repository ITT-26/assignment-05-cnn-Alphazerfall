import os
import sys
import argparse

# Must be set before cv2 import on Windows (MSMF backend)
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# Suppress TF/MediaPipe C++ log spam
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

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


# ---------- CLI ----------
parser = argparse.ArgumentParser(description="Gesture-controlled selfie camera")
parser.add_argument("--camera", type=int, default=0, help="Camera device ID")
parser.add_argument("--time", type=float, default=3.0, help="Countdown duration in seconds")
parser.add_argument("--path", type=str, default="selfie.jpg", help="File path to save captured image")
args = parser.parse_args()


# ---------- Constants ----------
UPDATE_HZ = 30
HAND_MODEL = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
GESTURE_MODEL = os.path.join(os.path.dirname(__file__), '..', '02-dataset', 'gesture_recognition.keras')
LABEL_NAMES = ['dislike', 'like', 'peace', 'rock', 'stop']
IMG_SIZE = 64
HAND_PAD = 0.15     # padding around MediaPipe bounding box as a fraction of box size
CONF_THRESH = 0.6   # minimum prediction confidence to report a gesture
GESTURE_STREAK = 4  # frames a gesture must be stable before it's accepted
DEBUG_HAND = False  # show extracted hand crop in a separate window (toggle: D)

# Color palette (BGR)
COL_BG_PANEL = (40, 40, 40)
COL_ACCENT = (90, 200, 255)
COL_TEXT = (255, 255, 255)
COL_TEXT_DARK = (30, 30, 30)
COL_OK = (80, 220, 120)
COL_WARN = (40, 40, 200)


# ---------- Camera thread ----------
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


# ---------- App states ----------
STATE_IDLE = 'idle'
STATE_COUNTDOWN = 'countdown'
STATE_CAPTURED = 'captured'


# ---------- Helpers ----------
def px(v):
    return max(1, int(round(v * SCALE)))


# ---------- Drawing ----------
def draw_border(frame, color, thickness=8):
    cv2.rectangle(frame, (0, 0), (WINDOW_WIDTH - 1, WINDOW_HEIGHT - 1),
                  color, px(thickness))


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
            label = f"{gesture}  {conf:.0%}"
            label_y = max(y1 - px(10), px(20))
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, SCALE * 0.7,
                        COL_TEXT_DARK, px(4), cv2.LINE_AA)
            cv2.putText(frame, label, (x1, label_y),
                        cv2.FONT_HERSHEY_DUPLEX, SCALE * 0.7,
                        COL_OK, px(2), cv2.LINE_AA)


def draw_hud(frame, gesture, app_state, countdown):
    bar_h = px(50)
    draw_panel(frame, (0, WINDOW_HEIGHT - bar_h), (WINDOW_WIDTH, WINDOW_HEIGHT))

    gesture_label = f"Gesture: {gesture}" if gesture else "Gesture: —"
    draw_text_centered(frame, gesture_label,
                       (WINDOW_WIDTH // 4, WINDOW_HEIGHT - bar_h // 2),
                       SCALE * 0.6, COL_TEXT, px(1), outline=False)

    if app_state == STATE_COUNTDOWN:
        state_label = f"Countdown: {countdown:.1f}s"
    else:
        state_label = f"State: {app_state}"
    draw_text_centered(frame, state_label,
                       (3 * WINDOW_WIDTH // 4, WINDOW_HEIGHT - bar_h // 2),
                       SCALE * 0.6, COL_ACCENT, px(1), outline=False)


# ---------- Hand detection ----------
def detect_hands(frame_bgr):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    ts = int(time.monotonic() * 1000)
    result = _landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    if not result.hand_landmarks:
        return []

    out = []
    for hand_landmarks in result.hand_landmarks:
        xs = [lm.x * w for lm in hand_landmarks]
        ys = [lm.y * h for lm in hand_landmarks]

        pad_x = int((max(xs) - min(xs)) * HAND_PAD)
        pad_y = int((max(ys) - min(ys)) * HAND_PAD)

        x1 = max(0, int(min(xs)) - pad_x)
        y1 = max(0, int(min(ys)) - pad_y)
        x2 = min(w, int(max(xs)) + pad_x)
        y2 = min(h, int(max(ys)) + pad_y)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            out.append((crop, (x1, y1, x2, y2)))

    return out


# ---------- Gesture classification ----------
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


# ---------- Main loop ----------
app = {
    'state': STATE_IDLE,
    'countdown': args.time,
    'streak_label': None,
    'streak_count': 0,
}


def update(dt):
    frame = cap.read()
    if frame is None:
        return

    display = cv2.flip(frame, 1)  # mirror for selfie view

    # --- Hand detection & gesture classification ---
    hands = detect_hands(display)
    if hands:
        with _infer_lock:
            if not _infer['busy']:
                _infer['busy'] = True
                threading.Thread(target=_infer_worker, args=(hands[0][0].copy(),), daemon=True).start()
        with _infer_lock:
            raw_label, raw_conf = _infer['cache']
    else:
        with _infer_lock:
            _infer['cache'] = (None, None)
        raw_label, raw_conf = None, None

    # Require the same gesture for GESTURE_STREAK consecutive frames
    if raw_label == app['streak_label']:
        app['streak_count'] += 1
    else:
        app['streak_label'] = raw_label
        app['streak_count'] = 1
    gesture = raw_label if app['streak_count'] >= GESTURE_STREAK else None
    conf = raw_conf if gesture else None

    if DEBUG_HAND:
        if hands:
            tile_h = 200
            tiles = [cv2.resize(c, (int(c.shape[1] * tile_h / c.shape[0]), tile_h)) for c, _ in hands]
            cv2.imshow('hand crop', np.hstack(tiles))
        else:
            cv2.imshow('hand crop', np.zeros((200, 200, 3), np.uint8))
        cv2.waitKey(1)

    # --- State machine ---
    if app['state'] == STATE_IDLE:
        pass  # TODO: trigger on gesture

    elif app['state'] == STATE_COUNTDOWN:
        app['countdown'] -= dt
        if app['countdown'] <= 0:
            cv2.imwrite(args.path, display)
            print(f"Saved to {args.path}")
            app['state'] = STATE_CAPTURED
            app['countdown'] = args.time

    elif app['state'] == STATE_CAPTURED:
        pass  # TODO: dismiss back to idle

    # --- Draw ---
    border_color = {
        STATE_IDLE: COL_ACCENT,
        STATE_COUNTDOWN: COL_WARN,
        STATE_CAPTURED: COL_OK,
    }.get(app['state'], COL_ACCENT)
    draw_border(display, border_color)

    if app['state'] == STATE_COUNTDOWN:
        draw_text_centered(display, str(int(app['countdown']) + 1),
                           (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2),
                           SCALE * 5.0, COL_WARN, px(6))

    draw_hand_overlay(display, hands, gesture, conf)
    draw_hud(display, gesture, app['state'], app['countdown'])

    img = pyglet.image.ImageData(
        WINDOW_WIDTH, WINDOW_HEIGHT, 'BGR',
        display.tobytes(), pitch=-WINDOW_WIDTH * 3,
    )
    _display_tex.blit_into(img, 0, 0, 0)


# ---------- Setup ----------
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

if not os.path.exists(HAND_MODEL):
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)

_landmarker = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
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

window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT, caption="Selfie Camera")
_display_tex = pyglet.image.Texture.create(WINDOW_WIDTH, WINDOW_HEIGHT)
display_sprite = pyglet.sprite.Sprite(_display_tex)

pyglet.clock.schedule_interval(update, 1 / UPDATE_HZ)


@window.event
def on_draw():
    window.clear()
    display_sprite.draw()


@window.event
def on_key_press(symbol, modifiers):
    global DEBUG_HAND
    if symbol == pyglet.window.key.Q:
        window.close()
    elif symbol == pyglet.window.key.D:
        DEBUG_HAND = not DEBUG_HAND
        if not DEBUG_HAND:
            cv2.destroyWindow('hand crop')


@window.event
def on_close():
    cap.release()
    _landmarker.close()


pyglet.app.run()