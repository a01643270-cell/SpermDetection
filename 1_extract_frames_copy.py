import cv2
from pathlib import Path
from tqdm import tqdm
import json


class FrameExtractor:
    def __init__(self, input_dir, output_dir, num_frames_per_video=100, image_format='jpg'):
        """
        Extract a specific number of frames from each video.

        Args:
            input_dir: Directory containing AVI files
            output_dir: Directory to save extracted frames
            num_frames_per_video: Exact number of frames to extract per video
            image_format: 'jpg' or 'png'
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_frames_per_video = num_frames_per_video
        self.image_format = image_format
        self.metadata = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_target_indices(self, total_frames):
        """
        Build evenly spaced frame indices without reading the whole video.
        """
        n = min(self.num_frames_per_video, total_frames)

        if n <= 1:
            return [0] if total_frames > 0 else []

        indices = [
            int(round(i * (total_frames - 1) / (n - 1)))
            for i in range(n)
        ]

        # Remove duplicates while preserving order
        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        return unique_indices

    def extract_from_video(self, video_path):
        """Extract only the selected frames from a single video."""
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return False

        video_name = video_path.stem
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"❌ Could not read total frame count for: {video_path}")
            cap.release()
            return False

        target_indices = self._build_target_indices(total_frames)

        video_output_dir = self.output_dir / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        video_metadata = {
            'video_name': video_name,
            'fps': fps,
            'total_frames': total_frames,
            'requested_frames': self.num_frames_per_video,
            'saved_frames': []
        }

        pbar = tqdm(total=len(target_indices), desc=f"Processing {video_name}")

        for frame_idx in target_indices:
            # Jump directly to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"⚠️ Could not read frame {frame_idx} from {video_name}")
                continue

            timestamp = frame_idx / fps if fps > 0 else None
            frame_filename = f"{video_name}_frame_{frame_idx:06d}.{self.image_format}"
            frame_path = video_output_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)

            video_metadata['saved_frames'].append({
                'frame_number': frame_idx,
                'timestamp': round(timestamp, 3) if timestamp is not None else None,
                'filename': frame_filename,
                'path': str(frame_path)
            })

            saved_count += 1
            pbar.update(1)

        pbar.close()
        cap.release()

        self.metadata[video_name] = video_metadata
        print(f"✅ Extracted {saved_count} frames from {video_name}")
        return True

    def extract_all(self):
        """Extract frames from all AVI files in input directory."""
        avi_files = list(self.input_dir.glob('*.avi'))

        if not avi_files:
            print(f"❌ No AVI files found in {self.input_dir}")
            return

        print(f"🎬 Found {len(avi_files)} videos")

        for video_path in avi_files:
            self.extract_from_video(video_path)

        metadata_file = self.output_dir / 'extraction_metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        print(f"\n✅ All videos processed!")
        print(f"📁 Frames saved to: {self.output_dir}")
        print(f"📊 Metadata saved to: {metadata_file}")


if __name__ == "__main__":
    INPUT_DIR = "datasets/raw_videos"
    OUTPUT_DIR = "datasets/frames"
    N_FRAMES = 5  # exact number of frames per video

    extractor = FrameExtractor(INPUT_DIR, OUTPUT_DIR, num_frames_per_video=N_FRAMES)
    extractor.extract_all()