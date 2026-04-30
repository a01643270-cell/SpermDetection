# -*- coding: utf-8 -*-
import sys
import io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
except AttributeError:
    pass

import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


class SpermTracker:
    def __init__(
        self,
        model_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        max_history=50,
    ):
        """
        Initialize YOLO model and DeepSort tracker.

        Args:
            model_path: Path to YOLO model weights (.pt)
            device: 'cuda' or 'cpu'
            max_history: Maximum trajectory points stored per track
        """
        self.device = device
        self.max_history = max_history
        print(f"🔧 Using device: {self.device}")

        # Load YOLO model
        print("📦 Loading YOLO model...")
        self.yolo = YOLO(model_path)

        # Initialize DeepSort tracker
        print("🎯 Initializing DeepSort tracker...")
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            nn_budget=100,
            embedder="mobilenet",
            half=(self.device == "cuda"),
            bgr=True,
            embedder_gpu=torch.cuda.is_available(),
        )

        self.track_history = defaultdict(list)
        self.detection_history = []

    @staticmethod
    def _xyxy_to_tlwh(x1, y1, x2, y2):
        """Convert x1,y1,x2,y2 to x,y,w,h for DeepSort."""
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return [x1, y1, w, h]

    def detect_and_track(self, frame, conf_threshold=0.5):
        """
        Detect objects and track them.

        Returns:
            annotated_frame, tracks, detections
        """
        results = self.yolo.predict(
            source=frame,
            conf=conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                tlwh = self._xyxy_to_tlwh(x1, y1, x2, y2)

                # Ignore tiny boxes
                if tlwh[2] < 5 or tlwh[3] < 5:
                    continue

                # DeepSort expects: ([x, y, w, h], confidence, class_name)
                detections.append((tlwh, conf, "sperm"))

                self.detection_history.append({
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": conf,
                })

        tracks = self.tracker.update_tracks(detections, frame=frame) if detections else []
        annotated_frame = frame.copy()

        confirmed_tracks = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            confirmed_tracks += 1
            track_id = track.track_id

            if hasattr(track, "to_ltrb"):
                bbox = track.to_ltrb()
            else:
                bbox = track.to_tlbr()

            x1, y1, x2, y2 = map(int, bbox)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.track_history[track_id].append((center_x, center_y))

            if len(self.track_history[track_id]) > self.max_history:
                self.track_history[track_id].pop(0)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated_frame,
                f"ID: {track_id}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            pts = self.track_history[track_id]
            for i in range(len(pts) - 1):
                cv2.line(annotated_frame, pts[i], pts[i + 1], (0, 150, 255), 1)

        info_text = f"Detections: {len(detections)} | Tracked: {confirmed_tracks}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return annotated_frame, tracks, detections

    def process_video(self, video_path, output_path=None, conf_threshold=0.5, skip_frames=1):
        """Process entire video with detection and tracking."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n🎬 Processing video: {Path(video_path).name}")
        print(f"   Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = 0
        processed_frames = 0
        all_tracks = set()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                annotated_frame, tracks, detections = self.detect_and_track(frame, conf_threshold)

                for track in tracks:
                    if track.is_confirmed():
                        all_tracks.add(track.track_id)

                if writer is not None:
                    writer.write(annotated_frame)

                processed_frames += 1

                if processed_frames % 30 == 0:
                    print(f"✅ Processed {processed_frames}/{total_frames} frames")

            frame_count += 1

        cap.release()
        if writer is not None:
            writer.release()

        print(f"\n✅ Video processing complete!")
        print(f"   Total frames: {total_frames}")
        print(f"   Processed frames: {processed_frames}")
        print(f"   Unique tracks: {len(all_tracks)}")
        print(f"   Total detections: {len(self.detection_history)}")

        if output_path:
            print(f"   Output video: {output_path}")

        return all_tracks

    def process_image(self, image_path, output_path=None, conf_threshold=0.5):
        """Process single image with detection and tracking."""
        frame = cv2.imread(str(image_path))

        if frame is None:
            print(f"❌ Error reading image: {image_path}")
            return None

        annotated_frame, tracks, detections = self.detect_and_track(frame, conf_threshold)

        print(f"✅ Image processed: {len(detections)} detections")

        if output_path is not None:
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"   Output image: {output_path}")

        return annotated_frame, tracks, detections


def _init_tk():
    """Create and immediately hide a Tk root window."""
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    return root


def _select_model_file():
    """Ask the user to select a YOLO model .pt file."""
    print("\n" + "=" * 60)
    print("🤖  STEP 1: Select YOLO model file  (.pt)")
    print("=" * 60)
    print("Please choose the trained YOLO model weights file (.pt).")

    default_model = Path("runs/detect/sperm_detector/weights/best.pt")
    default_dir = (
        str(default_model.parent.resolve())
        if default_model.parent.exists()
        else "."
    )

    root = _init_tk()
    model_file = filedialog.askopenfilename(
        title="Select YOLO model file (.pt)",
        initialdir=default_dir,
        filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
    )
    root.destroy()

    if not model_file:
        print("❌  No model file selected. Exiting.")
        return None

    model_path = Path(model_file)
    if not model_path.exists():
        print(f"❌  Model file not found: {model_path}")
        return None

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"✅  Model file : {model_path}")
    print(f"    Size       : {size_mb:.1f} MB")
    return str(model_path)


def _select_video_files():
    """Ask the user to select one or more input video files."""
    print("\n" + "=" * 60)
    print("🎬  STEP 2: Select input video file(s)")
    print("=" * 60)
    print("Please choose the video file(s) to process.")
    print("(You can select multiple files for batch processing.)")

    default_dir = (
        str(Path("datasets/raw_videos").resolve())
        if Path("datasets/raw_videos").exists()
        else "."
    )

    root = _init_tk()
    video_files = filedialog.askopenfilenames(
        title="Select input video file(s) — AVI, MP4, etc.",
        initialdir=default_dir,
        filetypes=[
            ("Video files", "*.avi *.mp4 *.mkv *.mov *.wmv"),
            ("AVI files", "*.avi"),
            ("MP4 files", "*.mp4"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not video_files:
        print("❌  No video file selected. Exiting.")
        return None

    videos = [Path(v) for v in video_files]
    print(f"✅  Selected {len(videos)} video(s):")
    for v in videos:
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"    📹 {v.name}  ({size_mb:.1f} MB)")
    return videos


def _select_output_folder():
    """Ask the user to select an output folder for results."""
    print("\n" + "=" * 60)
    print("📁  STEP 3: Select OUTPUT folder")
    print("=" * 60)
    print("Please choose the folder where result videos will be saved.")

    default = (
        str(Path("outputs").resolve())
        if Path("outputs").exists()
        else "."
    )

    root = _init_tk()
    folder = filedialog.askdirectory(
        title="Select output folder — processed videos will be saved here",
        initialdir=default,
    )
    root.destroy()

    if not folder:
        folder = "outputs"
        print(f"⚠️   No folder selected. Using default: {folder}")
    else:
        print(f"✅  Output folder : {folder}")

    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def _ask_processing_params():
    """Ask the user for detection confidence and frame skip."""
    print("\n" + "=" * 60)
    print("⚙️   STEP 4: Configure processing parameters")
    print("=" * 60)

    root = _init_tk()
    conf_raw = simpledialog.askfloat(
        "Detection Confidence",
        "Detection confidence threshold  (0.0 – 1.0)\n\n"
        "Higher = fewer but more certain detections.\nDefault: 0.5",
        initialvalue=0.5,
        minvalue=0.01,
        maxvalue=1.0,
        parent=root,
    )
    root.destroy()

    if conf_raw is None:
        conf_raw = 0.5
        print(f"⚠️   Using default confidence: {conf_raw}")
    else:
        print(f"✅  Confidence threshold : {conf_raw}")

    root2 = _init_tk()
    skip_raw = simpledialog.askinteger(
        "Frame Skip",
        "Process every N-th frame  (1 = all frames, 2 = every other, …)\n\n"
        "Higher values are faster but may miss fast movements.\nDefault: 1",
        initialvalue=1,
        minvalue=1,
        maxvalue=100,
        parent=root2,
    )
    root2.destroy()

    if skip_raw is None:
        skip_raw = 1
        print(f"⚠️   Using default skip: {skip_raw}")
    else:
        print(f"✅  Frame skip           : {skip_raw}")

    return conf_raw, skip_raw


def main():
    print("\n" + "🔬 " * 20)
    print("  SPERM DETECTION — YOLO + DeepSort Tracker")
    print("🔬 " * 20)

    model_path = _select_model_file()
    if model_path is None:
        raise SystemExit(1)

    video_files = _select_video_files()
    if video_files is None:
        raise SystemExit(1)

    output_dir = _select_output_folder()
    conf_threshold, skip_frames = _ask_processing_params()

    print("\n" + "=" * 60)
    print("📋  Configuration Summary")
    print("=" * 60)
    print(f"  🤖 Model           : {model_path}")
    print(f"  🎬 Videos          : {len(video_files)} file(s)")
    print(f"  📂 Output folder   : {output_dir}")
    print(f"  🎯 Confidence      : {conf_threshold}")
    print(f"  ⏭️   Frame skip      : {skip_frames}")
    print("=" * 60)

    root = _init_tk()
    proceed = messagebox.askyesno(
        "Confirm",
        f"Ready to process {len(video_files)} video(s).\n\n"
        f"Model     : {Path(model_path).name}\n"
        f"Output    : {output_dir}\n"
        f"Confidence: {conf_threshold}\n"
        f"Skip      : every {skip_frames} frame(s)\n\n"
        "Proceed?",
    )
    root.destroy()

    if not proceed:
        print("❌  Cancelled by user.")
        raise SystemExit(0)

    print("\n🚀  Loading model and starting processing...\n")
    tracker = SpermTracker(model_path=model_path)

    for video_path in video_files:
        output_name = f"{video_path.stem}_tracked.mp4"
        output_path = Path(output_dir) / output_name

        print(f"\n{'─' * 60}")
        print(f"📹  Processing: {video_path.name}")
        print(f"    → Output  : {output_path}")

        tracker.process_video(
            video_path,
            output_path=output_path,
            conf_threshold=conf_threshold,
            skip_frames=skip_frames,
        )

    print(f"\n{'=' * 60}")
    print(f"✅  All done! Results saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()