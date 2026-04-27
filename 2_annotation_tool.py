import cv2
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple


@dataclass
class BBox:
    """Bounding box annotation in ORIGINAL image coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_name: str = "sperm"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_display(self, scale: float) -> "BBox":
        """Convert original-image bbox to display-image bbox."""
        if scale <= 0:
            scale = 1.0
        return BBox(
            x_min=int(round(self.x_min * scale)),
            y_min=int(round(self.y_min * scale)),
            x_max=int(round(self.x_max * scale)),
            y_max=int(round(self.y_max * scale)),
            class_name=self.class_name
        )

    @staticmethod
    def from_display_points(
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
        scale: float,
        class_name: str = "sperm"
    ) -> "BBox":
        """Convert two display-image points into an ORIGINAL-image bbox."""
        if scale <= 0:
            scale = 1.0

        x1, y1 = start_pt
        x2, y2 = end_pt
        inv = 1.0 / scale

        x_min = int(round(min(x1, x2) * inv))
        y_min = int(round(min(y1, y2) * inv))
        x_max = int(round(max(x1, x2) * inv))
        y_max = int(round(max(y1, y2) * inv))

        return BBox(x_min, y_min, x_max, y_max, class_name)


class AnnotationTool:
    def __init__(self, images_dir, annotations_dir="datasets/annotations"):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        # Search recursively so it finds images inside subfolders too
        self.image_files = sorted(
            [
                f for f in self.images_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ],
            key=lambda p: str(p).lower()
        )

        self.current_idx = 0
        self.current_image_path: Optional[Path] = None

        # Original image (full resolution)
        self.original_image = None

        # Display image (possibly resized)
        self.display_image = None
        self.display_scale = 1.0

        self.current_bboxes: List[BBox] = []

        self.drawing = False
        self.start_point_display: Optional[Tuple[int, int]] = None
        self.preview_point_display: Optional[Tuple[int, int]] = None

        print(f"📁 Found {len(self.image_files)} images to annotate")

    def _annotation_file_for(self, image_path: Path) -> Path:
        # Keeps the same “one JSON per image” style
        return self.annotations_dir / f"{image_path.stem}_annotations.json"

    def load_annotations(self, image_path: Path) -> List[BBox]:
        """Load existing annotations for an image."""
        annotation_file = self._annotation_file_for(image_path)

        if annotation_file.exists():
            try:
                with open(annotation_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [BBox(**bbox) for bbox in data.get("bboxes", [])]
            except Exception as e:
                print(f"⚠️ Could not load annotations for {image_path.name}: {e}")
                return []

        return []

    def save_annotations(self, image_path: Path):
        """Save current annotations to JSON."""
        annotation_file = self._annotation_file_for(image_path)

        data = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "num_bboxes": len(self.current_bboxes),
            "bboxes": [bbox.to_dict() for bbox in self.current_bboxes],
        }

        with open(annotation_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ Saved {len(self.current_bboxes)} annotations for {image_path.name}")

    def _prepare_display_image(self, image):
        """Resize image only for display, keeping original intact."""
        height, width = image.shape[:2]
        max_w, max_h = 1920, 1080

        scale = min(1.0, max_w / width, max_h / height)

        if scale < 1.0:
            disp = cv2.resize(
                image,
                (int(round(width * scale)), int(round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            disp = image.copy()

        return disp, scale

    def _clamp_bbox_to_image(self, bbox: BBox) -> BBox:
        """Clamp bbox to original-image bounds."""
        if self.original_image is None:
            return bbox

        h, w = self.original_image.shape[:2]

        x_min = max(0, min(w - 1, bbox.x_min))
        y_min = max(0, min(h - 1, bbox.y_min))
        x_max = max(0, min(w - 1, bbox.x_max))
        y_max = max(0, min(h - 1, bbox.y_max))

        # Ensure correct ordering after clamping
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)

        return BBox(x_min, y_min, x_max, y_max, bbox.class_name)

    def redraw_all_bboxes(self, preview_end: Optional[Tuple[int, int]] = None):
        """Redraw all saved bboxes plus the current preview box."""
        if self.original_image is None:
            return

        self.display_image = self.display_image_base.copy()

        # Draw saved annotations
        for i, bbox in enumerate(self.current_bboxes):
            db = bbox.to_display(self.display_scale)

            cv2.rectangle(
                self.display_image,
                (db.x_min, db.y_min),
                (db.x_max, db.y_max),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                self.display_image,
                f"#{i + 1}",
                (db.x_min, max(15, db.y_min - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        # Draw the active preview rectangle while dragging
        if self.drawing and self.start_point_display is not None and preview_end is not None:
            cv2.rectangle(
                self.display_image,
                self.start_point_display,
                preview_end,
                (0, 255, 255),
                2,
            )

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bboxes."""
        if self.original_image is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point_display = (x, y)
            self.preview_point_display = (x, y)
            self.redraw_all_bboxes(preview_end=(x, y))

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.preview_point_display = (x, y)
                self.redraw_all_bboxes(preview_end=(x, y))

        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point_display is not None:
                self.drawing = False

                end_point = (x, y)
                bbox = BBox.from_display_points(
                    self.start_point_display,
                    end_point,
                    self.display_scale,
                    class_name="sperm",
                )
                bbox = self._clamp_bbox_to_image(bbox)

                # Minimum bbox size in ORIGINAL coordinates
                if (bbox.x_max - bbox.x_min) > 10 and (bbox.y_max - bbox.y_min) > 10:
                    self.current_bboxes.append(bbox)
                    print(f"✏️  Added bbox #{len(self.current_bboxes)}")
                else:
                    print("⚠️  Bbox too small, ignored")

                self.start_point_display = None
                self.preview_point_display = None
                self.redraw_all_bboxes()

    def show_instructions(self):
        """Print usage instructions."""
        print("\n" + "=" * 60)
        print("🖱️  ANNOTATION TOOL - KEYBOARD SHORTCUTS")
        print("=" * 60)
        print("LEFT CLICK + DRAG   : Draw bounding box around sperm")
        print("U                   : Undo last annotation")
        print("S                   : Save annotations for current image")
        print("D                   : Delete all annotations (current image)")
        print("N or RIGHT ARROW    : Next image")
        print("P or LEFT ARROW     : Previous image")
        print("SPACE               : Jump to image number")
        print("Q or ESC            : Quit")
        print("=" * 60 + "\n")

    def _load_current_image(self, image_path: Path) -> bool:
        """Load original image and prepare the display version."""
        self.original_image = cv2.imread(str(image_path))
        if self.original_image is None:
            return False

        self.display_image_base, self.display_scale = self._prepare_display_image(self.original_image)
        self.display_image = self.display_image_base.copy()
        return True

    def annotate(self):
        """Main annotation loop."""
        self.show_instructions()

        window_name = "Sperm Annotation Tool"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while self.current_idx < len(self.image_files):
            image_path = self.image_files[self.current_idx]
            self.current_image_path = image_path

            print(f"\n📸 Image {self.current_idx + 1}/{len(self.image_files)}: {image_path.name}")

            if not self._load_current_image(image_path):
                print(f"❌ Could not read image: {image_path}")
                self.current_idx += 1
                continue

            # Load existing annotations
            self.current_bboxes = self.load_annotations(image_path)

            self.drawing = False
            self.start_point_display = None
            self.preview_point_display = None

            self.redraw_all_bboxes()
            print(f"Current annotations: {len(self.current_bboxes)}")

            while True:
                display = self.display_image.copy()
                text = f"Image {self.current_idx + 1}/{len(self.image_files)} | Bboxes: {len(self.current_bboxes)}"
                cv2.putText(
                    display,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.imshow(window_name, display)

                # Use waitKeyEx to support arrow keys more reliably
                key = cv2.waitKeyEx(1)
                key_low = key & 0xFF

                RIGHT_KEYS = {83, 2555904, 65363}
                LEFT_KEYS = {81, 2424832, 65361}

                if key_low in (ord("q"), 27):  # Q or ESC
                    self.save_annotations(image_path)
                    cv2.destroyAllWindows()
                    return

                elif key_low == ord("n") or key in RIGHT_KEYS:
                    self.save_annotations(image_path)
                    self.current_idx += 1
                    break

                elif key_low == ord("p") or key in LEFT_KEYS:
                    self.save_annotations(image_path)
                    self.current_idx = max(0, self.current_idx - 1)
                    break

                elif key_low == ord("u"):  # Undo
                    if self.current_bboxes:
                        self.current_bboxes.pop()
                        print(f"↩️  Undone. Remaining: {len(self.current_bboxes)}")
                        self.redraw_all_bboxes()

                elif key_low == ord("s"):  # Save
                    self.save_annotations(image_path)

                elif key_low == ord("d"):  # Delete all
                    self.current_bboxes = []
                    print("🗑️  All annotations cleared")
                    self.redraw_all_bboxes()

                elif key_low == ord(" "):  # Space: jump to image
                    self.save_annotations(image_path)
                    cv2.destroyAllWindows()

                    try:
                        img_num = int(input(f"Enter image number (1-{len(self.image_files)}): "))
                        self.current_idx = max(0, min(img_num - 1, len(self.image_files) - 1))
                    except ValueError:
                        print("⚠️  Invalid number. Staying on current image.")

                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(window_name, self.mouse_callback)
                    break


if __name__ == "__main__":
    IMAGES_DIR = "datasets/frames"         # Directory with extracted frames
    ANNOTATIONS_DIR = "datasets/annotations"  # Directory to save annotations

    tool = AnnotationTool(IMAGES_DIR, ANNOTATIONS_DIR)
    tool.annotate()