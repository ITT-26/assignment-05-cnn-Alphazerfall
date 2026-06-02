#!/usr/bin/env python3
"""HaGRID-compatible bounding box annotation tool. See README.md for usage."""

import cv2
import json
import argparse
from pathlib import Path

# 1-9 -> indices 0-8, 0 -> index 9, N -> no_gesture
ALL_LABELS = ["like", "dislike", "stop", "rock", "peace",
              "fist", "ok", "one", "three", "two_up", "no_gesture"]
COLORS = {
    "like":        (0, 220, 0),
    "dislike":     (0, 0, 220),
    "stop":        (220, 180, 0),
    "rock":        (200, 0, 200),
    "peace":       (0, 200, 220),
    "fist":        (0, 140, 255),
    "ok":          (60, 220, 180),
    "one":         (255, 80, 80),
    "three":       (180, 255, 60),
    "two_up":      (255, 200, 0),
    "no_gesture":  (120, 120, 120),
}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Maximum display size, image is scaled down to fit if larger (but not scaled up if smaller)
DISPLAY_MAX_W = 1000  # 1600
DISPLAY_MAX_H = 800  # 1100


def collect_images(root: Path) -> list[tuple[Path, str]]:
    """Return (image_path, gesture_label) pairs from gesture subfolders."""
    pairs = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or folder.name.startswith("_"):
            continue
        for img in sorted(folder.iterdir()):
            if img.suffix in IMG_EXTS:
                pairs.append((img, folder.name))
    return pairs


def load_annotations(source: Path) -> dict:
    """Load annotations from a single JSON file or a directory of JSON files."""
    annotations = {}
    if source.is_dir():
        for json_file in sorted(source.glob("*.json")):
            with open(json_file) as f:
                annotations.update(json.load(f))
        print(f"Loaded {len(annotations)} annotations from {source}/")
    elif source.is_file() and source.stat().st_size > 0:
        with open(source) as f:
            annotations = json.load(f)
        print(f"Loaded {len(annotations)} annotations from {source}")
    return annotations


class Annotator:
    def __init__(self, images_root: Path, output_file: Path,
                 load_source: Path | None = None, readonly: bool = False):
        self.output_file = output_file
        self.readonly = readonly
        self.annotations: dict = {}

        filter_to_annotated = False
        if load_source:
            self.annotations = load_annotations(load_source)
            filter_to_annotated = load_source.is_file()  # single JSON -> only show those images
        elif output_file.exists() and output_file.stat().st_size > 0:
            self.annotations = load_annotations(output_file)

        self.images = collect_images(images_root)
        if filter_to_annotated:
            self.images = [(p, l) for p, l in self.images if p.stem in self.annotations]
        if not self.images:
            raise SystemExit(f"No images found under gesture subfolders in {images_root}")
        print(f"Found {len(self.images)} images across "
              f"{len({lbl for _, lbl in self.images})} gesture(s).")

        # Original image dimensions
        self.img_w = self.img_h = 1
        # Display image (scaled) and scale factor (disp / orig)
        self.disp: "cv2.Mat" = None
        self.disp_w = self.disp_h = 1
        self.scale = 1.0

        self.folder_label = ALL_LABELS[0]
        self.current_label = ALL_LABELS[0]
        self.bboxes: list = []
        self.box_labels: list = []

        self.drawing = False
        self.sx = self.sy = self.ex = self.ey = 0

    # ------------------------------------------------------------------
    def _make_display(self, img):
        """Scale img to fit DISPLAY_MAX_W × DISPLAY_MAX_H, store scale."""
        h, w = img.shape[:2]
        self.img_w, self.img_h = w, h
        scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
        self.scale = scale
        if scale < 1.0:
            self.disp_w = int(w * scale)
            self.disp_h = int(h * scale)
            self.disp = cv2.resize(img, (self.disp_w, self.disp_h), interpolation=cv2.INTER_AREA)
        else:
            self.disp_w, self.disp_h = w, h
            self.disp = img.copy()

    def _save(self):
        if self.readonly:
            return
        with open(self.output_file, "w") as f:
            json.dump(self.annotations, f, indent=4)

    def _commit(self, img_id: str):
        if self.readonly or not self.bboxes:
            return
        self.annotations[img_id] = {
            "bboxes": self.bboxes,
            "labels": self.box_labels,
        }

    # ------------------------------------------------------------------
    def _redraw(self):
        """Render boxes + live rubber-band + HUD onto a fresh copy of disp."""
        frame = self.disp.copy()

        # Existing confirmed boxes
        for bbox, label in zip(self.bboxes, self.box_labels):
            x, y, w, h = bbox
            # bbox is normalised to ORIGINAL dims -> convert to display dims
            x1 = int(x * self.img_w * self.scale)
            y1 = int(y * self.img_h * self.scale)
            x2 = int((x + w) * self.img_w * self.scale)
            y2 = int((y + h) * self.img_h * self.scale)
            color = COLORS.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Live rubber-band rect while dragging
        if self.drawing:
            color = COLORS.get(self.current_label, (255, 255, 255))
            cv2.rectangle(frame, (self.sx, self.sy), (self.ex, self.ey), color, 2)

        # Top HUD: label selector (two rows) with current label highlighted
        keys = [str(i + 1) for i in range(9)] + ["0", "N"]
        parts = []
        for key, lbl in zip(keys, ALL_LABELS):
            tag = f"[{key}]{lbl}"
            if lbl == self.current_label:
                tag = f">>>{tag}<<<"
            parts.append(tag)
        row1 = "  ".join(parts[:6])
        row2 = "  ".join(parts[6:])
        cv2.rectangle(frame, (0, 0), (self.disp_w, 52), (30, 30, 30), -1)
        cv2.putText(frame, row1, (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)
        cv2.putText(frame, row2, (6, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1)

        # Bottom HUD: controls
        cv2.rectangle(frame, (0, self.disp_h - 28), (self.disp_w, self.disp_h),
                      (30, 30, 30), -1)
        controls = ("S=next   Q=quit  [READ-ONLY]"
                    if self.readonly else
                    "Z=undo   S=save+next   Q=save+quit")
        cv2.putText(frame, controls, (6, self.disp_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        cv2.imshow("Annotator", frame)

    # ------------------------------------------------------------------
    def _mouse(self, event, x, y, *_):
        if self.readonly:
            return
        # x, y are in display-pixel space (because the window size == disp size)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.sx, self.sy = x, y
            self.ex, self.ey = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.ex, self.ey = x, y
            self._redraw()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            x1, y1 = min(self.sx, x), min(self.sy, y)
            x2, y2 = max(self.sx, x), max(self.sy, y)
            if (x2 - x1) > 8 and (y2 - y1) > 8:
                # Convert display pixels -> normalised original-image coords
                self.bboxes.append([
                    x1 / (self.img_w * self.scale),
                    y1 / (self.img_h * self.scale),
                    (x2 - x1) / (self.img_w * self.scale),
                    (y2 - y1) / (self.img_h * self.scale),
                ])
                self.box_labels.append(self.current_label)
                # Next box defaults to no_gesture
                self.current_label = "no_gesture"
            self._redraw()

    # ------------------------------------------------------------------
    def run(self):
        cv2.namedWindow("Annotator", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Annotator", self._mouse)

        total = len(self.images)
        for idx, (img_path, folder_label) in enumerate(self.images):
            img_id = img_path.stem
            raw = cv2.imread(str(img_path))
            if raw is None:
                print(f"Cannot read {img_path.name}, skipping.")
                continue

            self._make_display(raw)
            self.folder_label = folder_label

            if img_id in self.annotations:
                self.bboxes = [list(b) for b in self.annotations[img_id]["bboxes"]]
                self.box_labels = list(self.annotations[img_id]["labels"])
                self.current_label = self.box_labels[-1] if self.box_labels else folder_label
            else:
                self.bboxes = []
                self.box_labels = []
                self.current_label = folder_label

            print(f"\n[{idx+1}/{total}] {folder_label}/{img_path.name}  "
                  f"({self.img_w}x{self.img_h} → displayed {self.disp_w}x{self.disp_h})")
            self._redraw()

            while True:
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    continue

                if key == ord('q'):
                    self._commit(img_id)
                    self._save()
                    cv2.destroyAllWindows()
                    if self.readonly:
                        print("\nDone.")
                    else:
                        print(f"\nDone. {len(self.annotations)} images annotated → {self.output_file}")
                    return

                elif key == ord('s'):
                    self._commit(img_id)
                    self._save()
                    print(f"  Saved {len(self.bboxes)} box(es).")
                    break

                elif key == ord('z') and not self.readonly:
                    if self.bboxes:
                        self.bboxes.pop()
                        self.box_labels.pop()
                        self.current_label = (
                            self.box_labels[-1] if self.box_labels else self.folder_label
                        )
                        self._redraw()

                elif key == ord('n') and not self.readonly:
                    self.current_label = "no_gesture"
                    print(f"  Label → no_gesture")
                    self._redraw()

                elif not self.readonly:
                    if ord('1') <= key <= ord('9'):
                        i = key - ord('1')
                    elif key == ord('0'):
                        i = 9
                    else:
                        i = -1
                    if 0 <= i < len(ALL_LABELS) - 1:  # exclude no_gesture from number keys
                        self.current_label = ALL_LABELS[i]
                        print(f"  Label → {self.current_label}")
                        self._redraw()

        self._save()
        cv2.destroyAllWindows()
        if not self.readonly:
            print(f"\nAll images annotated. {len(self.annotations)} entries → {self.output_file}")


def main():
    parser = argparse.ArgumentParser(description="HaGRID-compatible annotation tool")
    parser.add_argument("--images", required=True,
                        help="Root folder with gesture subfolders (like/, dislike/, ...)")
    parser.add_argument("--output", default="annot-name.json",
                        help="Output JSON file (default: annot-name.json)")
    parser.add_argument("--load",
                        help="Load annotations from a JSON file or a directory of JSON files "
                             "(e.g. ../data/_annotations/). Overrides --output for loading.")
    parser.add_argument("--readonly", action="store_true",
                        help="View-only mode: annotations are shown but cannot be edited.")
    args = parser.parse_args()

    root = Path(args.images)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    load_source = Path(args.load) if args.load else None
    Annotator(root, Path(args.output), load_source, args.readonly).run()


if __name__ == "__main__":
    main()
