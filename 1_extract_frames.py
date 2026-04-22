import cv2
import os
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

class FrameExtractor:
    def __init__(self, input_dir, output_dir, frame_interval=1, image_format='jpg'):
        """
        Extract frames from AVI videos
        
        Args:
            input_dir: Directory containing AVI files
            output_dir: Directory to save extracted frames
            frame_interval: Extract every nth frame (1 = all frames)
            image_format: 'jpg' or 'png'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.frame_interval = frame_interval
        self.image_format = image_format
        self.metadata = {}
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_from_video(self, video_path):
        """Extract frames from a single video"""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return False
        
        video_name = video_path.stem
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video-specific output directory
        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        video_metadata = {
            'video_name': video_name,
            'fps': fps,
            'total_frames': total_frames,
            'extracted_frames': []
        }
        
        pbar = tqdm(total=total_frames, desc=f"Processing {video_name}")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract every nth frame
            if frame_count % self.frame_interval == 0:
                timestamp = frame_count / fps
                frame_filename = f"{video_name}_frame_{frame_count:06d}.{self.image_format}"
                frame_path = video_output_dir / frame_filename
                
                # Save frame
                cv2.imwrite(str(frame_path), frame)
                
                video_metadata['extracted_frames'].append({
                    'frame_number': frame_count,
                    'timestamp': round(timestamp, 3),
                    'filename': frame_filename,
                    'path': str(frame_path)
                })
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        
        self.metadata[video_name] = video_metadata
        print(f"✅ Extracted {saved_count} frames from {video_name}")
        return True
    
    def extract_all(self):
        """Extract frames from all AVI files in input directory"""
        avi_files = list(self.input_dir.glob('*.avi'))
        
        if not avi_files:
            print(f"❌ No AVI files found in {self.input_dir}")
            return
        
        print(f"🎬 Found {len(avi_files)} videos")
        
        for video_path in avi_files:
            self.extract_from_video(video_path)
        
        # Save metadata
        metadata_file = self.output_dir / 'extraction_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"\n✅ All videos processed!")
        print(f"📁 Frames saved to: {self.output_dir}")
        print(f"📊 Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "raw_videos"      # Directory with AVI files
    OUTPUT_DIR = "datasets/frames" # Directory to save frames
    FRAME_INTERVAL = 1             # Extract every frame (1), every 2nd frame (2), etc.
    
    # Create extractor and process videos
    extractor = FrameExtractor(INPUT_DIR, OUTPUT_DIR, frame_interval=FRAME_INTERVAL)
    extractor.extract_all()