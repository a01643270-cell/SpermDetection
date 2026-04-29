import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict


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


def main():
    BASE_DIR = Path(r"C:\Projects\SpermDetection")

    MODEL_PATH = BASE_DIR / "runs" / "detect" / "runs" / "sperm_yolo" / "exp1-3" / "weights" / "best.pt"
    INPUT_VIDEO = BASE_DIR / "datasets" / "raw_videos" / "test_video.avi"
    OUTPUT_VIDEO = BASE_DIR / "outputs" / "tracked_video.mp4"

    CONF_THRESHOLD = 0.1
    SKIP_FRAMES = 1

    if not MODEL_PATH.exists():
        print(f"❌ Model not found: {MODEL_PATH}")
        return

    if not INPUT_VIDEO.exists():
        print(f"❌ Input video not found: {INPUT_VIDEO}")
        return

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)

    tracker = SpermTracker(model_path=str(MODEL_PATH))

    tracker.process_video(
        INPUT_VIDEO,
        output_path=OUTPUT_VIDEO,
        conf_threshold=CONF_THRESHOLD,
        skip_frames=SKIP_FRAMES,
    )


if __name__ == "__main__":
    main()