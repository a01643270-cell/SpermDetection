import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import json
from datetime import datetime

class SpermTracker:
    def __init__(self, model_path="yolov8s.pt", device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize YOLO model and DeepSort tracker
        
        Args:
            model_path: Path to YOLO model weights
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"🔧 Using device: {self.device}")
        
        # Load YOLO model
        print("📦 Loading YOLO model...")
        self.yolo = YOLO(model_path)
        self.yolo.to(self.device)
        
        # Initialize DeepSort tracker
        print("🎯 Initializing DeepSort tracker...")
        self.tracker = DeepSort(
            model_name="osnet_x0_25",
            device=self.device,
            max_dist=0.2,
            max_iou_distance=0.7,
            max_age=30,
            n_init=3,
            nn_budget=100,
            embedder_gpu=torch.cuda.is_available()
        )
        
        self.track_history = defaultdict(list)
        self.detection_history = []
        
    def detect_and_track(self, frame, conf_threshold=0.5):
        """Detect sperm and track them
        
        Args:
            frame: Input frame (numpy array)
            conf_threshold: Confidence threshold for detection
            
        Returns:
            frame: Annotated frame
            tracks: Track information
        """
        # Run YOLO detection
        results = self.yolo(frame, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        bboxes = []
        
        # Process detections
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                # Prepare detection for DeepSort
                bbox = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                bboxes.append(bbox[0])
                detections.append(([x1, y1, x2, y2], conf, "sperm"))
                
                self.detection_history.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf
                })
        
        # Update tracker
        tracks = self.tracker.update_tracks(bboxes, frame=frame) if bboxes else []
        
        # Draw annotations
        annotated_frame = frame.copy()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_tlbr()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Store track history
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.track_history[track_id].append((center_x, center_y))
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw track ID
            label = f"ID: {track_id}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw trajectory
            if len(self.track_history[track_id]) > 1:
                for i in range(len(self.track_history[track_id]) - 1):
                    p1 = self.track_history[track_id][i]
                    p2 = self.track_history[track_id][i + 1]
                    cv2.line(annotated_frame, p1, p2, (0, 150, 255), 1)
        
        # Draw statistics
        info_text = f"Detections: {len(detections)} | Tracked: {len(tracks)}"
        cv2.putText(annotated_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return annotated_frame, tracks, detections
    
    def process_video(self, video_path, output_path=None, conf_threshold=0.5, skip_frames=1):
        """Process entire video with detection and tracking
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            conf_threshold: Detection confidence threshold
            skip_frames: Process every nth frame (1 = all frames)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n🎬 Processing video: {Path(video_path).name}")
        print(f"   Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output requested
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        processed_frames = 0
        all_tracks = set()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process every skip_frames frame
            if frame_count % skip_frames == 0:
                annotated_frame, tracks, detections = self.detect_and_track(frame, conf_threshold)
                
                # Track unique IDs
                for track in tracks:
                    if track.is_confirmed():
                        all_tracks.add(track.track_id)
                
                if writer:
                    writer.write(annotated_frame)
                
                processed_frames += 1
                
                if processed_frames % 30 == 0:
                    print(f"✅ Processed {processed_frames}/{total_frames} frames")
            
            frame_count += 1
        
        cap.release()
        if writer:
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
        """Process single image with detection and tracking
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            conf_threshold: Detection confidence threshold
        """
        frame = cv2.imread(str(image_path))
        
        if frame is None:
            print(f"❌ Error reading image: {image_path}")
            return
        
        annotated_frame, tracks, detections = self.detect_and_track(frame, conf_threshold)
        
        print(f"✅ Image processed: {len(detections)} detections")
        
        if output_path:
            cv2.imwrite(str(output_path), annotated_frame)
            print(f"   Output image: {output_path}")
        
        return annotated_frame, tracks, detections

def main():
    """Main entry point"""
    
    # Configuration
    MODEL_PATH = "yolov8s.pt"  # Pre-trained model or path to trained model
    INPUT_VIDEO = "path/to/video.avi"  # Input video file
    OUTPUT_VIDEO = "output_tracked.mp4"  # Output video file
    CONF_THRESHOLD = 0.5
    SKIP_FRAMES = 1  # Process every frame (1), every 2nd frame (2), etc.
    
    # Initialize tracker
    tracker = SpermTracker(model_path=MODEL_PATH)
    
    # Process video
    if Path(INPUT_VIDEO).exists():
        all_tracks = tracker.process_video(
            INPUT_VIDEO, 
            output_path=OUTPUT_VIDEO,
            conf_threshold=CONF_THRESHOLD,
            skip_frames=SKIP_FRAMES
        )
    else:
        print(f"❌ Input video not found: {INPUT_VIDEO}")

if __name__ == "__main__":
    main()
