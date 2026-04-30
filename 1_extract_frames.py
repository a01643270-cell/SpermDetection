# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import cv2
from pathlib import Path
from tqdm import tqdm
import json
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


class FrameExtractor:
    def __init__(self, input_dir, output_dir, num_frames_per_video=100, image_format='jpg'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.num_frames_per_video = num_frames_per_video
        self.image_format = image_format
        self.metadata = {}

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_target_indices(self, total_frames):
        n = min(self.num_frames_per_video, total_frames)

        if n <= 1:
            return [0] if total_frames > 0 else []

        indices = [
            int(round(i * (total_frames - 1) / (n - 1)))
            for i in range(n)
        ]

        seen = set()
        unique_indices = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                unique_indices.append(idx)

        return unique_indices

    def extract_from_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print(f"❌ Error opening video: {video_path}")
            return False

        relative_video = video_path.relative_to(self.input_dir).with_suffix('')
        video_key = "_".join(relative_video.parts)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            print(f"❌ Could not read total frame count for: {video_path}")
            cap.release()
            return False

        target_indices = self._build_target_indices(total_frames)

        saved_count = 0
        video_metadata = {
            'video_name': str(relative_video),
            'fps': fps,
            'total_frames': total_frames,
            'requested_frames': self.num_frames_per_video,
            'saved_frames': []
        }

        pbar = tqdm(total=len(target_indices), desc=f"Processing {relative_video}")

        for frame_idx in target_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                print(f"⚠️ Could not read frame {frame_idx} from {video_path}")
                continue

            timestamp = frame_idx / fps if fps > 0 else None

            # Todos los frames en una sola carpeta
            frame_filename = f"{video_key}_frame_{frame_idx:06d}.{self.image_format}"
            frame_path = self.output_dir / frame_filename

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

        self.metadata[video_key] = video_metadata
        print(f"✅ Extracted {saved_count} frames from {relative_video}")
        return True

    def extract_all(self):
        """Busca AVI en la carpeta seleccionada y en todas sus subcarpetas."""
        avi_files = list(self.input_dir.rglob('*.avi')) + list(self.input_dir.rglob('*.AVI'))

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


def _init_tk():
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    return root


def _select_input_folder():
    print("\n" + "=" * 60)
    print("📁  STEP 1: Select INPUT folder  (raw_videos)")
    print("=" * 60)
    print("Please choose the folder that contains your .avi video files.")

    default = (
        str(Path("datasets/raw_videos").resolve())
        if Path("datasets/raw_videos").exists()
        else "."
    )

    root = _init_tk()
    folder = filedialog.askdirectory(
        title="Select folder with .avi videos",
        initialdir=default,
    )
    root.destroy()

    if not folder:
        print("❌  No folder selected. Exiting.")
        return None

    folder_path = Path(folder)
    avi_files = list(folder_path.rglob("*.avi")) + list(folder_path.rglob("*.AVI"))

    print(f"✅  Input folder: {folder}")
    print(f"    Found {len(avi_files)} .avi file(s) including subfolders")

    return folder


def _select_output_folder():
    print("\n" + "=" * 60)
    print("📁  STEP 2: Select OUTPUT folder  (frames)")
    print("=" * 60)
    print("Please choose the folder where extracted frames will be saved.")
    print("(All frames will be saved in this single folder.)")

    default = (
        str(Path("datasets/frames").resolve())
        if Path("datasets/frames").exists()
        else "."
    )

    root = _init_tk()
    folder = filedialog.askdirectory(
        title="Select output folder",
        initialdir=default,
    )
    root.destroy()

    if not folder:
        folder = "datasets/frames"
        print(f"⚠️ Using default: {folder}")
    else:
        print(f"✅ Output folder: {folder}")

    return folder


def _ask_num_frames():
    print("\n" + "=" * 60)
    print("⚙️   STEP 3: Configure extraction parameters")
    print("=" * 60)

    root = _init_tk()
    result = simpledialog.askinteger(
        "Frames per Video",
        "How many frames should be extracted from each video?\n\nDefault: 5",
        initialvalue=5,
        minvalue=1,
        maxvalue=10000,
        parent=root,
    )
    root.destroy()

    if result is None:
        result = 5
        print(f"⚠️   No value entered. Using default: {result} frames per video.")
    else:
        print(f"✅  Frames per video: {result}")

    return result


if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("  SPERM DETECTION — Frame Extractor")
    print("🔬 " * 20)

    input_dir = _select_input_folder()
    if input_dir is None:
        raise SystemExit(1)

    output_dir = _select_output_folder()
    n_frames = _ask_num_frames()

    print("\n" + "=" * 60)
    print("📋  Configuration Summary")
    print("=" * 60)
    print(f"  📂 Input  (raw_videos) : {input_dir}")
    print(f"  📂 Output (frames)     : {output_dir}")
    print(f"  🎞️   Frames per video   : {n_frames}")
    print("=" * 60)

    root = _init_tk()
    proceed = messagebox.askyesno(
        "Confirm",
        f"Ready to extract frames.\n\n"
        f"Input  : {input_dir}\n"
        f"Output : {output_dir}\n"
        f"Frames per video: {n_frames}\n\n"
        "Proceed?",
    )
    root.destroy()

    if not proceed:
        print("❌  Cancelled by user.")
        raise SystemExit(0)

    print("\n🚀  Starting extraction...\n")
    extractor = FrameExtractor(input_dir, output_dir, num_frames_per_video=n_frames)
    extractor.extract_all()