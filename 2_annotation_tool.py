#!/usr/bin/env python3
"""Interactive OpenCV annotation tool for sperm bounding boxes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MIN_BOX_SIZE = 3
TEXT_LINE_HEIGHT = 28


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int = 0

    def as_dict(self) -> dict:
        return {
            "x1": int(min(self.x1, self.x2)),
            "y1": int(min(self.y1, self.y2)),
            "x2": int(max(self.x1, self.x2)),
            "y2": int(max(self.y1, self.y2)),
            "class_id": self.class_id,
        }


class AnnotationTool:
    def __init__(self, images_dir: Path, annotations_dir: Path, window_name: str = "Sperm Annotation Tool"):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.window_name = window_name

        self.image_paths = sorted(
            [p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {images_dir}")

        self.idx = 0
        self.boxes: list[BBox] = []
        self.current_image = None
        self.current_path: Optional[Path] = None

        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.tmp_x = 0
        self.tmp_y = 0

        self.annotations_dir.mkdir(parents=True, exist_ok=True)

    def rel_image_path(self, image_path: Path) -> Path:
        return image_path.relative_to(self.images_dir)

    def annotation_path(self, image_path: Path) -> Path:
        return (self.annotations_dir / self.rel_image_path(image_path)).with_suffix(".json")

    def load_image_and_annotations(self) -> None:
        self.current_path = self.image_paths[self.idx]
        self.current_image = cv2.imread(str(self.current_path))
        if self.current_image is None:
            raise RuntimeError(f"Could not read image: {self.current_path}")

        ann_path = self.annotation_path(self.current_path)
        self.boxes = []
        if ann_path.exists():
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            for b in data.get("bboxes", []):
                self.boxes.append(BBox(b["x1"], b["y1"], b["x2"], b["y2"], int(b.get("class_id", 0))))

    def save_annotations(self) -> None:
        if self.current_path is None or self.current_image is None:
            return

        h, w = self.current_image.shape[:2]
        ann_path = self.annotation_path(self.current_path)
        ann_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "image": str(self.rel_image_path(self.current_path)).replace("\\", "/"),
            "width": int(w),
            "height": int(h),
            "bboxes": [b.as_dict() for b in self.boxes],
        }
        ann_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def delete_annotation_file(self) -> None:
        if self.current_path is None:
            return
        ann_path = self.annotation_path(self.current_path)
        if ann_path.exists():
            ann_path.unlink()

    def render_canvas(self):
        if self.current_image is None:
            return None

        canvas = self.current_image.copy()

        for i, box in enumerate(self.boxes, start=1):
            b = box.as_dict()
            cv2.rectangle(canvas, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 255, 0), 2)
            cv2.putText(
                canvas,
                f"id:{i}",
                (b["x1"], max(20, b["y1"] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if self.drawing:
            cv2.rectangle(canvas, (self.start_x, self.start_y), (self.tmp_x, self.tmp_y), (0, 165, 255), 2)

        help_lines = [
            f"Image {self.idx + 1}/{len(self.image_paths)}: {self.current_path.name if self.current_path else ''}",
            f"Boxes: {len(self.boxes)}",
            "Mouse: drag left button to create box | Right click: undo",
            "Keys: [N/Space] next  [P] prev  [S] save  [U] undo  [D] delete file  [Q] quit",
        ]

        y = TEXT_LINE_HEIGHT
        for line in help_lines:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 1, cv2.LINE_AA)
            y += TEXT_LINE_HEIGHT

        return canvas

    def draw_ui(self) -> None:
        canvas = self.render_canvas()
        if canvas is not None:
            cv2.imshow(self.window_name, canvas)

    def handle_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x, self.start_y = x, y
            self.tmp_x, self.tmp_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.tmp_x, self.tmp_y = x, y
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.tmp_x, self.tmp_y = x, y
            if abs(self.start_x - x) >= MIN_BOX_SIZE and abs(self.start_y - y) >= MIN_BOX_SIZE:
                self.boxes.append(BBox(self.start_x, self.start_y, x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.boxes:
                self.boxes.pop()

    def run(self, screenshot_path: Optional[Path] = None) -> None:
        if screenshot_path is not None:
            self.load_image_and_annotations()
            canvas = self.render_canvas()
            if canvas is None:
                raise RuntimeError("Unable to render annotation canvas")
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(screenshot_path), canvas)
            print(f"Screenshot saved to: {screenshot_path.resolve()}")
            return

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.handle_mouse)

        while True:
            self.load_image_and_annotations()

            while True:
                self.draw_ui()
                key = cv2.waitKey(16) & 0xFF

                if key in (ord("q"), 27):
                    self.save_annotations()
                    cv2.destroyAllWindows()
                    return
                if key in (ord("s"),):
                    self.save_annotations()
                elif key in (ord("u"),):
                    if self.boxes:
                        self.boxes.pop()
                elif key in (ord("d"),):
                    self.boxes = []
                    self.delete_annotation_file()
                elif key in (ord("n"), 32):
                    self.save_annotations()
                    if self.idx >= len(self.image_paths) - 1:
                        print("Already at last image.")
                        continue
                    self.idx += 1
                    break
                elif key in (ord("p"),):
                    self.save_annotations()
                    if self.idx <= 0:
                        print("Already at first image.")
                        continue
                    self.idx -= 1
                    break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive annotation tool")
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/frames"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("datasets/annotations"))
    parser.add_argument(
        "--screenshot-path",
        type=Path,
        default=None,
        help="Save one UI screenshot and exit (useful for headless validation)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tool = AnnotationTool(images_dir=args.images_dir, annotations_dir=args.annotations_dir)
    tool.run(screenshot_path=args.screenshot_path)


if __name__ == "__main__":
    main()
