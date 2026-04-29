import cv2
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple
from PIL import Image, ImageTk


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

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
            class_name=self.class_name,
        )

    @staticmethod
    def from_display_points(
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
        scale: float,
        class_name: str = "sperm",
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


# ---------------------------------------------------------------------------
# GUI Application
# ---------------------------------------------------------------------------

class AnnotationTool:
    """Tkinter-based annotation tool with modern UI and folder-selection dialogs."""

    # Color palette
    BG_COLOR   = "#1e1e2e"
    TOOLBAR_BG = "#2d2d3f"
    BTN_BG     = "#3d3d5c"
    BTN_HOVER  = "#5d5d8c"
    BTN_ACTIVE = "#7d7dbc"
    BTN_FG     = "#ffffff"
    STATUS_BG  = "#13131f"
    STATUS_FG  = "#a0a0c0"
    ACCENT     = "#7c6af7"
    SUCCESS    = "#50fa7b"
    WARNING    = "#ffb86c"
    ERROR      = "#ff5555"
    CANVAS_BG  = "#0d0d1a"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Sperm Annotation Tool")
        self.root.configure(bg=self.BG_COLOR)
        self.root.resizable(True, True)

        # Internal state
        self.images_dir: Optional[Path] = None
        self.annotations_dir: Optional[Path] = None
        self.image_files: List[Path] = []
        self.current_idx: int = 0

        self.original_image = None          # RGB numpy array (full resolution)
        self.display_scale: float = 1.0
        self.current_bboxes: List[BBox] = []

        self.drawing: bool = False
        self.start_point: Optional[Tuple[int, int]] = None  # display coords

        self._saved: bool = True
        self._tk_image = None               # must be kept alive to avoid GC
        self._canvas_w: int = 800
        self._canvas_h: int = 600
        self._img_x_off: int = 0            # image offset inside canvas
        self._img_y_off: int = 0

        self._build_ui()
        # Prompt the user to pick folders as soon as the window appears
        self.root.after(100, self._select_folders)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_progress_row()
        self._build_canvas()
        self._build_status_bar()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _build_toolbar(self) -> None:
        toolbar = tk.Frame(self.root, bg=self.TOOLBAR_BG, pady=8)
        toolbar.pack(fill=tk.X, side=tk.TOP)

        # (label, key hint, command)
        button_specs = [
            ("⬅  Prev",    "[P / ←]",  self.prev_image),
            ("Next  ➡",   "[N / →]",  self.next_image),
            ("↩  Undo",    "[U]",       self.undo_annotation),
            ("💾  Save",   "[S]",       self._save_current),
            ("🗑  Clear",  "[D]",       self.clear_annotations),
            ("🔢  Jump",   "[Space]",   self.jump_to_image),
            ("📁  Folders","",          self._select_folders),
        ]

        for label, hint, cmd in button_specs:
            self._make_toolbar_button(toolbar, label, hint, cmd)

        # Help text aligned to the right
        tk.Label(
            toolbar,
            text="DRAG to draw box  |  Q / ESC to quit",
            bg=self.TOOLBAR_BG,
            fg=self.STATUS_FG,
            font=("Segoe UI", 9),
        ).pack(side=tk.RIGHT, padx=16)

    def _make_toolbar_button(
        self, parent: tk.Frame, label: str, hint: str, cmd
    ) -> tk.Button:
        frame = tk.Frame(parent, bg=self.TOOLBAR_BG)
        frame.pack(side=tk.LEFT, padx=5)

        btn = tk.Button(
            frame,
            text=label,
            command=cmd,
            bg=self.BTN_BG,
            fg=self.BTN_FG,
            font=("Segoe UI", 9, "bold"),
            relief=tk.FLAT,
            padx=12,
            pady=6,
            cursor="hand2",
            bd=0,
        )
        btn.pack()

        if hint:
            tk.Label(
                frame,
                text=hint,
                bg=self.TOOLBAR_BG,
                fg=self.STATUS_FG,
                font=("Segoe UI", 7),
            ).pack()

        btn.bind("<Enter>",          lambda e, b=btn: b.configure(bg=self.BTN_HOVER))
        btn.bind("<Leave>",          lambda e, b=btn: b.configure(bg=self.BTN_BG))
        btn.bind("<ButtonPress-1>",  lambda e, b=btn: b.configure(bg=self.BTN_ACTIVE))
        btn.bind("<ButtonRelease-1>",lambda e, b=btn: b.configure(bg=self.BTN_HOVER))

        return btn

    def _build_progress_row(self) -> None:
        row = tk.Frame(self.root, bg=self.BG_COLOR, pady=4)
        row.pack(fill=tk.X, side=tk.TOP)

        self._progress_label = tk.Label(
            row,
            text="No images loaded",
            bg=self.BG_COLOR,
            fg=self.STATUS_FG,
            font=("Segoe UI", 9),
        )
        self._progress_label.pack(side=tk.LEFT, padx=12)

        self._progress_bar = ttk.Progressbar(
            row, orient=tk.HORIZONTAL, mode="determinate", length=300
        )
        self._progress_bar.pack(side=tk.LEFT, padx=8)

        self._save_state_label = tk.Label(
            row,
            text="● Saved",
            bg=self.BG_COLOR,
            fg=self.SUCCESS,
            font=("Segoe UI", 9),
        )
        self._save_state_label.pack(side=tk.RIGHT, padx=8)

        self._anno_count_label = tk.Label(
            row,
            text="Annotations: 0",
            bg=self.BG_COLOR,
            fg=self.ACCENT,
            font=("Segoe UI", 9, "bold"),
        )
        self._anno_count_label.pack(side=tk.RIGHT, padx=12)

    def _build_canvas(self) -> None:
        canvas_frame = tk.Frame(self.root, bg=self.BG_COLOR)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.canvas = tk.Canvas(
            canvas_frame,
            bg=self.CANVAS_BG,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>",       self._on_canvas_resize)
        self.canvas.bind("<ButtonPress-1>",   self._on_mouse_press)
        self.canvas.bind("<B1-Motion>",       self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_release)

    def _build_status_bar(self) -> None:
        bar = tk.Frame(self.root, bg=self.STATUS_BG, pady=4)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_label = tk.Label(
            bar,
            text="Ready — select a folder to start annotating.",
            bg=self.STATUS_BG,
            fg=self.STATUS_FG,
            font=("Segoe UI", 9),
            anchor=tk.W,
        )
        self._status_label.pack(side=tk.LEFT, padx=12)

    def _bind_keys(self) -> None:
        for key in ("<n>", "<N>", "<Right>"):
            self.root.bind(key, lambda e: self.next_image())
        for key in ("<p>", "<P>", "<Left>"):
            self.root.bind(key, lambda e: self.prev_image())
        for key in ("<u>", "<U>"):
            self.root.bind(key, lambda e: self.undo_annotation())
        for key in ("<s>", "<S>"):
            self.root.bind(key, lambda e: self._save_current())
        for key in ("<d>", "<D>"):
            self.root.bind(key, lambda e: self.clear_annotations())
        self.root.bind("<space>",  lambda e: self.jump_to_image())
        for key in ("<q>", "<Q>", "<Escape>"):
            self.root.bind(key, lambda e: self._quit())

    # ── Status / progress helpers ──────────────────────────────────────────

    def _set_status(self, msg: str, color: Optional[str] = None) -> None:
        self._status_label.configure(text=msg, fg=color or self.STATUS_FG)

    def _set_save_state(self, saved: bool) -> None:
        self._saved = saved
        if saved:
            self._save_state_label.configure(text="● Saved",   fg=self.SUCCESS)
        else:
            self._save_state_label.configure(text="● Unsaved", fg=self.WARNING)

    def _update_progress(self) -> None:
        if not self.image_files:
            self._progress_label.configure(text="No images loaded")
            self._progress_bar["value"] = 0
            self._anno_count_label.configure(text="Annotations: 0")
            return

        total   = len(self.image_files)
        current = self.current_idx + 1
        name    = self.image_files[self.current_idx].name

        self._progress_label.configure(
            text=f"Image {current} / {total}:  {name}"
        )
        self._progress_bar["maximum"] = total
        self._progress_bar["value"]   = current
        self._anno_count_label.configure(
            text=f"Annotations: {len(self.current_bboxes)}"
        )

    # ── Folder selection ───────────────────────────────────────────────────

    def _select_folders(self) -> None:
        """Open dialogs so the user can pick images and annotations folders."""
        default_images = (
            str(Path("datasets/frames").resolve())
            if Path("datasets/frames").exists()
            else "."
        )
        images_dir = filedialog.askdirectory(
            title="Select Images Folder (extracted frames)",
            initialdir=default_images,
        )
        if not images_dir:
            if self.images_dir is None:
                self._set_status(
                    "No folder selected — press '📁 Folders' to choose one.",
                    self.WARNING,
                )
            return

        default_anno = (
            str(Path("datasets/annotations").resolve())
            if Path("datasets/annotations").exists()
            else images_dir
        )
        annotations_dir = filedialog.askdirectory(
            title="Select Annotations Output Folder",
            initialdir=default_anno,
        )
        if not annotations_dir:
            annotations_dir = str(Path(images_dir) / "annotations")

        self.images_dir      = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)

        self.image_files = sorted(
            [
                f for f in self.images_dir.rglob("*")
                if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
            ],
            key=lambda p: str(p).lower(),
        )

        if not self.image_files:
            messagebox.showwarning(
                "No Images Found",
                f"No JPG/PNG images found in:\n{self.images_dir}",
            )
            self._set_status(f"No images found in {self.images_dir}", self.ERROR)
            return

        self.current_idx = 0
        self._set_status(
            f"Loaded {len(self.image_files)} images from '{self.images_dir.name}'  |  "
            f"Annotations → '{self.annotations_dir.name}'"
        )
        self._load_image(self.current_idx)

    # ── Annotation I/O ─────────────────────────────────────────────────────

    def _annotation_file_for(self, image_path: Path) -> Path:
        return self.annotations_dir / f"{image_path.stem}_annotations.json"

    def load_annotations(self, image_path: Path) -> List[BBox]:
        """Load existing annotations for an image."""
        anno_file = self._annotation_file_for(image_path)
        if anno_file.exists():
            try:
                with open(anno_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return [BBox(**b) for b in data.get("bboxes", [])]
            except Exception as exc:
                self._set_status(f"Could not load annotations: {exc}", self.ERROR)
        return []

    def save_annotations(self, image_path: Path) -> None:
        """Save current annotations to JSON (same format as before)."""
        if image_path is None or self.annotations_dir is None:
            return

        anno_file = self._annotation_file_for(image_path)
        data = {
            "image_name": image_path.name,
            "image_path": str(image_path),
            "num_bboxes": len(self.current_bboxes),
            "bboxes":     [b.to_dict() for b in self.current_bboxes],
        }
        with open(anno_file, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        self._set_save_state(True)
        self._set_status(
            f"Saved {len(self.current_bboxes)} annotations for {image_path.name}",
            self.SUCCESS,
        )

    def _save_current(self) -> None:
        if self.image_files and self.current_idx < len(self.image_files):
            self.save_annotations(self.image_files[self.current_idx])

    # ── Image loading & display ────────────────────────────────────────────

    def _load_image(self, idx: int) -> None:
        """Load the image at *idx* and refresh the canvas."""
        if not self.image_files:
            return

        self.current_idx = max(0, min(idx, len(self.image_files) - 1))
        image_path = self.image_files[self.current_idx]

        raw = cv2.imread(str(image_path))
        if raw is None:
            self._set_status(f"Could not read image: {image_path.name}", self.ERROR)
            return

        self.original_image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self.current_bboxes = self.load_annotations(image_path)
        self.drawing     = False
        self.start_point = None

        self._set_save_state(True)
        self._update_progress()
        self._refresh_canvas()

    def _get_display_scale(self) -> float:
        if self.original_image is None:
            return 1.0
        h, w = self.original_image.shape[:2]
        cw = max(1, self._canvas_w)
        ch = max(1, self._canvas_h)
        return min(1.0, cw / w, ch / h)

    def _refresh_canvas(
        self, preview_end: Optional[Tuple[int, int]] = None
    ) -> None:
        """Redraw the canvas: image + saved boxes + optional preview box."""
        if self.original_image is None:
            return

        scale = self._get_display_scale()
        self.display_scale = scale
        h, w = self.original_image.shape[:2]
        dw, dh = int(round(w * scale)), int(round(h * scale))

        # Resize for display
        if scale < 1.0:
            disp = cv2.resize(
                self.original_image, (dw, dh), interpolation=cv2.INTER_AREA
            )
        else:
            disp = self.original_image.copy()

        # Draw saved bounding boxes
        for i, bbox in enumerate(self.current_bboxes):
            db = bbox.to_display(scale)
            cv2.rectangle(
                disp, (db.x_min, db.y_min), (db.x_max, db.y_max),
                (80, 250, 123), 2,
            )
            cv2.putText(
                disp, f"#{i + 1}",
                (db.x_min + 2, max(15, db.y_min - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 250, 123), 1,
            )

        # Draw live preview box while dragging
        if self.drawing and self.start_point and preview_end:
            cv2.rectangle(
                disp, self.start_point, preview_end,
                (255, 184, 108), 2,
            )

        # Push to tkinter canvas (centered)
        pil_img = Image.fromarray(disp)
        self._tk_image = ImageTk.PhotoImage(pil_img)

        x_off = max(0, (self._canvas_w - dw) // 2)
        y_off = max(0, (self._canvas_h - dh) // 2)
        self._img_x_off = x_off
        self._img_y_off = y_off

        self.canvas.delete("all")
        self.canvas.create_image(x_off, y_off, anchor=tk.NW, image=self._tk_image)

    # ── Canvas event handlers ──────────────────────────────────────────────

    def _canvas_to_img(self, cx: int, cy: int) -> Tuple[int, int]:
        """Translate canvas pixel coords to display-image pixel coords."""
        return cx - self._img_x_off, cy - self._img_y_off

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self._canvas_w = event.width
        self._canvas_h = event.height
        self._refresh_canvas()

    def _on_mouse_press(self, event: tk.Event) -> None:
        if self.original_image is None:
            return
        self.drawing = True
        self.start_point = self._canvas_to_img(event.x, event.y)
        self._refresh_canvas(preview_end=self.start_point)

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if self.drawing and self.original_image is not None:
            self._refresh_canvas(
                preview_end=self._canvas_to_img(event.x, event.y)
            )

    def _on_mouse_release(self, event: tk.Event) -> None:
        if (
            not self.drawing
            or self.start_point is None
            or self.original_image is None
        ):
            return

        self.drawing = False
        end_pt = self._canvas_to_img(event.x, event.y)

        bbox = BBox.from_display_points(
            self.start_point, end_pt, self.display_scale, class_name="sperm"
        )
        bbox = self._clamp_bbox_to_image(bbox)

        if (bbox.x_max - bbox.x_min) > 10 and (bbox.y_max - bbox.y_min) > 10:
            self.current_bboxes.append(bbox)
            self._set_save_state(False)
            self._set_status(f"Added box #{len(self.current_bboxes)}", self.ACCENT)
        else:
            self._set_status("Box too small — ignored.", self.WARNING)

        self.start_point = None
        self._refresh_canvas()
        self._update_progress()

    def _clamp_bbox_to_image(self, bbox: BBox) -> BBox:
        """Clamp bbox to original-image bounds."""
        if self.original_image is None:
            return bbox
        h, w = self.original_image.shape[:2]
        x_min = max(0, min(w - 1, bbox.x_min))
        y_min = max(0, min(h - 1, bbox.y_min))
        x_max = max(0, min(w - 1, bbox.x_max))
        y_max = max(0, min(h - 1, bbox.y_max))
        x_min, x_max = min(x_min, x_max), max(x_min, x_max)
        y_min, y_max = min(y_min, y_max), max(y_min, y_max)
        return BBox(x_min, y_min, x_max, y_max, bbox.class_name)

    # ── Navigation & editing actions ───────────────────────────────────────

    def next_image(self) -> None:
        if not self.image_files:
            return
        self._save_current()
        if self.current_idx < len(self.image_files) - 1:
            self._load_image(self.current_idx + 1)
        else:
            self._set_status("Already at the last image.", self.WARNING)

    def prev_image(self) -> None:
        if not self.image_files:
            return
        self._save_current()
        if self.current_idx > 0:
            self._load_image(self.current_idx - 1)
        else:
            self._set_status("Already at the first image.", self.WARNING)

    def undo_annotation(self) -> None:
        if self.current_bboxes:
            self.current_bboxes.pop()
            self._set_save_state(False)
            self._set_status(
                f"Undone — {len(self.current_bboxes)} annotation(s) remaining.",
                self.ACCENT,
            )
            self._refresh_canvas()
            self._update_progress()
        else:
            self._set_status("Nothing to undo.", self.WARNING)

    def clear_annotations(self) -> None:
        if not self.current_bboxes:
            self._set_status("No annotations to clear.", self.WARNING)
            return
        if messagebox.askyesno(
            "Clear All",
            f"Delete all {len(self.current_bboxes)} annotation(s) for this image?",
        ):
            self.current_bboxes = []
            self._set_save_state(False)
            self._set_status("All annotations cleared.", self.WARNING)
            self._refresh_canvas()
            self._update_progress()

    def jump_to_image(self) -> None:
        if not self.image_files:
            return
        self._save_current()

        dialog = tk.Toplevel(self.root)
        dialog.title("Jump to Image")
        dialog.configure(bg=self.BG_COLOR)
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog,
            text=f"Enter image number  (1 \u2013 {len(self.image_files)}):",
            bg=self.BG_COLOR,
            fg=self.BTN_FG,
            font=("Segoe UI", 10),
        ).pack(padx=20, pady=(20, 8))

        entry = tk.Entry(
            dialog, font=("Segoe UI", 11), width=10, justify=tk.CENTER
        )
        entry.pack(padx=20, pady=4)
        entry.focus_set()

        def _go() -> None:
            try:
                num = int(entry.get())
                idx = max(0, min(num - 1, len(self.image_files) - 1))
                dialog.destroy()
                self._load_image(idx)
            except ValueError:
                messagebox.showerror(
                    "Invalid Input", "Please enter a valid number.", parent=dialog
                )

        entry.bind("<Return>", lambda _e: _go())
        tk.Button(
            dialog,
            text="Go",
            command=_go,
            bg=self.ACCENT,
            fg=self.BTN_FG,
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            padx=20,
            pady=6,
            cursor="hand2",
        ).pack(padx=20, pady=(4, 20))

    def _quit(self) -> None:
        if not self._saved and self.image_files:
            if messagebox.askyesno(
                "Unsaved Changes",
                "You have unsaved annotations.\nSave before quitting?",
            ):
                self._save_current()
        self.root.destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x800")
    AnnotationTool(root)
    root.mainloop()