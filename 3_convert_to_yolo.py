#!/usr/bin/env python3
"""Convert JSON annotations from annotation tool to YOLO format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert annotations to YOLO dataset format")
    parser.add_argument("--images-dir", type=Path, default=Path("datasets/frames"))
    parser.add_argument("--annotations-dir", type=Path, default=Path("datasets/annotations"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/yolo_data"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-name", type=str, default="sperm")
    return parser.parse_args()


def bbox_to_yolo(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> tuple[float, float, float, float]:
    x1, x2 = sorted((max(0, x1), min(width, x2)))
    y1, y2 = sorted((max(0, y1), min(height, y2)))

    bw = max(0, x2 - x1)
    bh = max(0, y2 - y1)
    cx = x1 + bw / 2
    cy = y1 + bh / 2

    return cx / width, cy / height, bw / width, bh / height


def collect_images(images_dir: Path) -> list[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def split_items(items: list[Path], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[Path]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train/val/test ratios must sum to 1.0")

    random.Random(seed).shuffle(items)
    n = len(items)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val : n_train + n_val + n_test],
    }


def convert_one(image_path: Path, images_dir: Path, annotations_dir: Path) -> tuple[str, bool]:
    rel = image_path.relative_to(images_dir)
    ann_path = (annotations_dir / rel).with_suffix(".json")

    if not ann_path.exists():
        return "", False

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    width = int(data["width"])
    height = int(data["height"])

    lines = []
    for box in data.get("bboxes", []):
        x, y, w, h = bbox_to_yolo(
            int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]), width=width, height=height
        )
        if w <= 0 or h <= 0:
            continue
        class_id = int(box.get("class_id", 0))
        lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return "\n".join(lines), True


def main() -> None:
    args = parse_args()

    images = collect_images(args.images_dir)
    if not images:
        print(f"No images found in {args.images_dir.resolve()}")
        return

    splits = split_items(images, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)

    for split in ("train", "val", "test"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    converted = 0
    missing = 0

    for split, split_images in splits.items():
        for img_path in split_images:
            label_txt, has_ann = convert_one(img_path, args.images_dir, args.annotations_dir)
            if not has_ann:
                missing += 1
                continue

            target_img = args.output_dir / "images" / split / img_path.name
            target_label = args.output_dir / "labels" / split / f"{img_path.stem}.txt"
            shutil.copy2(img_path, target_img)
            target_label.write_text(label_txt + ("\n" if label_txt else ""), encoding="utf-8")
            converted += 1

    dataset_yaml = {
        "path": str(args.output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: args.class_name},
    }
    (args.output_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    print(f"Total images: {len(images)}")
    print(f"Converted annotations: {converted}")
    print(f"Missing annotations skipped: {missing}")
    print(f"Dataset config: {(args.output_dir / 'dataset.yaml').resolve()}")


if __name__ == "__main__":
    main()
