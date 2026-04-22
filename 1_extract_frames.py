#!/usr/bin/env python3
"""Extract frames from AVI videos for annotation/training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from AVI videos.")
    parser.add_argument("--input-dir", type=Path, default=Path("datasets/raw_videos"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/frames"))
    parser.add_argument("--frame-step", type=int, default=5, help="Save every Nth frame (>=1)")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality 0-100")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing extracted frames")
    return parser.parse_args()


def extract_video_frames(video_path: Path, output_dir: Path, frame_step: int, jpeg_quality: int, overwrite: bool) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for old in output_dir.glob("frame_*.jpg"):
            old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    saved = 0
    frame_idx = 0

    pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=video_path.name, unit="frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            saved += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    return {
        "video": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": frame_idx,
        "saved_frames": saved,
        "frame_step": frame_step,
    }


def main() -> None:
    args = parse_args()

    if args.frame_step < 1:
        raise ValueError("--frame-step must be >= 1")

    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.avi"))
    if not videos:
        print(f"No AVI videos found in {input_dir.resolve()}")
        return

    metadata_rows = []
    for video_path in videos:
        video_output_dir = output_dir / video_path.stem
        stats = extract_video_frames(
            video_path=video_path,
            output_dir=video_output_dir,
            frame_step=args.frame_step,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )
        metadata_rows.append(stats)

    metadata_path = output_dir / "extraction_metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video", "fps", "width", "height", "total_frames", "saved_frames", "frame_step"],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Processed {len(metadata_rows)} videos")
    print(f"Metadata saved to: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
