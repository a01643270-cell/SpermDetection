# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import yaml


class AnnotationConverter:
    def __init__(
        self,
        annotations_dir,
        images_dir,
        output_dir="datasets/yolo_data",
        seed=42,
        keep_empty_labels=True,
    ):
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.keep_empty_labels = keep_empty_labels

        # YOLO class list
        self.class_names = ["sperm"]

        # Build a fast lookup table for images, including subfolders
        self.image_index = self._build_image_index()

        # Create output directory structure
        self.setup_directories()

    def _build_image_index(self):
        """
        Build an index of all images found recursively under images_dir.
        Primary key: filename (e.g., video_frame_000123.jpg)
        """
        index = {}
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                # If there are duplicate names, keep the first one found
                index.setdefault(p.name, p)
        return index

    def setup_directories(self):
        """Create YOLO dataset directory structure."""
        splits = ["train", "val", "test"]
        subdirs = ["images", "labels"]

        for split in splits:
            for subdir in subdirs:
                path = self.output_dir / split / subdir
                path.mkdir(parents=True, exist_ok=True)

    def read_image_size(self, image_path: Path) -> Optional[Tuple[int, int]]:
        """Read image dimensions using OpenCV."""
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        height, width = img.shape[:2]
        return width, height

    def convert_bbox_to_yolo(self, bbox: dict, image_width: int, image_height: int) -> Optional[str]:
        """
        Convert bbox from pixel coordinates to YOLO format:
        <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
        """
        try:
            x_min = float(bbox["x_min"])
            y_min = float(bbox["y_min"])
            x_max = float(bbox["x_max"])
            y_max = float(bbox["y_max"])
        except (KeyError, TypeError, ValueError):
            return None

        # Basic validation
        if image_width <= 0 or image_height <= 0:
            return None

        if x_max <= x_min or y_max <= y_min:
            return None

        # Clamp to image bounds
        x_min = max(0.0, min(x_min, image_width - 1))
        y_min = max(0.0, min(y_min, image_height - 1))
        x_max = max(0.0, min(x_max, image_width - 1))
        y_max = max(0.0, min(y_max, image_height - 1))

        if x_max <= x_min or y_max <= y_min:
            return None

        x_center = ((x_min + x_max) / 2.0) / image_width
        y_center = ((y_min + y_max) / 2.0) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        # Final sanity clamp
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        width = max(0.0, min(1.0, width))
        height = max(0.0, min(1.0, height))

        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"

    def _find_image_for_annotation(self, data: dict) -> Optional[Path]:
        """
        Resolve the image path robustly:
        1) Prefer the stored image_path if it exists.
        2) Fall back to searching by image_name in the recursive image index.
        """
        image_path_str = data.get("image_path")
        if image_path_str:
            candidate = Path(image_path_str)
            if candidate.exists():
                return candidate

            # If image_path is relative, try resolving under images_dir
            candidate2 = self.images_dir / candidate.name
            if candidate2.exists():
                return candidate2

        image_name = data.get("image_name")
        if image_name:
            return self.image_index.get(image_name)

        return None

    def convert_annotations(self) -> List[Tuple[Path, Path]]:
        """
        Convert all annotations to YOLO format.
        Returns list of (image_path, label_path) tuples.
        """
        annotation_files = sorted(self.annotations_dir.glob("*_annotations.json"))
        dataset_pairs = []

        print(f"🔄 Converting {len(annotation_files)} annotation files...")

        temp_dir = self.output_dir / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        for annotation_file in annotation_files:
            try:
                with open(annotation_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"⚠️  Could not read annotation file {annotation_file.name}: {e}")
                continue

            image_path = self._find_image_for_annotation(data)
            if image_path is None or not image_path.exists():
                print(f"⚠️  Image not found for annotation: {annotation_file.name}")
                continue

            img_size = self.read_image_size(image_path)
            if img_size is None:
                print(f"⚠️  Could not read image: {image_path}")
                continue

            image_width, image_height = img_size

            yolo_labels = []
            for bbox in data.get("bboxes", []):
                yolo_line = self.convert_bbox_to_yolo(bbox, image_width, image_height)
                if yolo_line is not None:
                    yolo_labels.append(yolo_line)

            # Optional: skip images that ended up with no valid boxes
            if not self.keep_empty_labels and len(yolo_labels) == 0:
                continue

            label_filename = f"{image_path.stem}.txt"
            label_path = temp_dir / label_filename

            with open(label_path, "w", encoding="utf-8") as f:
                f.writelines(yolo_labels)

            dataset_pairs.append((image_path, label_path))

            if len(dataset_pairs) % 100 == 0:
                print(f"✅ Converted {len(dataset_pairs)} images")

        print(f"✅ Conversion complete: {len(dataset_pairs)} images")
        return dataset_pairs

    def split_dataset(
        self,
        dataset_pairs: List[Tuple[Path, Path]],
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
    ):
        """Split dataset into train/val/test and copy files."""
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        random.seed(self.seed)
        random.shuffle(dataset_pairs)

        total = len(dataset_pairs)
        train_idx = int(total * train_ratio)
        val_idx = int(total * (train_ratio + val_ratio))

        splits = {
            "train": dataset_pairs[:train_idx],
            "val": dataset_pairs[train_idx:val_idx],
            "test": dataset_pairs[val_idx:],
        }

        print(f"\n📊 Dataset split:")
        print(f"  Train: {len(splits['train'])} images ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(splits['val'])} images ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(splits['test'])} images ({test_ratio*100:.0f}%)")

        for split_name, pairs in splits.items():
            print(f"\n📁 Copying {split_name} images...")
            for img_path, label_path in pairs:
                dest_img = self.output_dir / split_name / "images" / img_path.name
                dest_label = self.output_dir / split_name / "labels" / label_path.name

                shutil.copy2(img_path, dest_img)
                shutil.copy2(label_path, dest_label)

            print(f"✅ {split_name} split copied")

        # Cleanup temp directory
        temp_dir = self.output_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training."""
        dataset_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": len(self.class_names),
            "names": self.class_names,
        }

        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(dataset_yaml, f, sort_keys=False, allow_unicode=True)

        print(f"\n✅ Created dataset.yaml at: {yaml_path}")
        return yaml_path

    def generate_statistics(self):
        """Generate dataset statistics."""
        print("\n" + "=" * 60)
        print("📊 DATASET STATISTICS")
        print("=" * 60)

        for split in ["train", "val", "test"]:
            labels_dir = self.output_dir / split / "labels"
            images_dir = self.output_dir / split / "images"

            label_files = list(labels_dir.glob("*.txt"))
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg")) + list(images_dir.glob("*.png"))

            total_objects = 0
            for label_file in label_files:
                with open(label_file, "r", encoding="utf-8") as f:
                    total_objects += sum(1 for _ in f)

            avg_objects = total_objects / len(label_files) if label_files else 0

            print(f"\n{split.upper()}:")
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            print(f"  Total objects: {total_objects}")
            print(f"  Avg objects per image: {avg_objects:.2f}")

    def convert(self):
        """Main conversion pipeline."""
        print("🚀 Starting YOLO annotation conversion...\n")

        dataset_pairs = self.convert_annotations()

        if not dataset_pairs:
            print("❌ No valid dataset pairs found!")
            return

        self.split_dataset(dataset_pairs)
        self.create_dataset_yaml()
        self.generate_statistics()

        print(f"\n✅ YOLO dataset ready at: {self.output_dir}")


def _init_tk():
    """Create and immediately hide a Tk root window."""
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    return root


def _select_annotations_folder():
    """Ask the user to select the annotations input folder."""
    print("\n" + "=" * 60)
    print("📁  STEP 1: Select ANNOTATIONS folder")
    print("=" * 60)
    print("Please choose the folder that contains the JSON annotation files.")

    default = (
        str(Path("datasets/annotations").resolve())
        if Path("datasets/annotations").exists()
        else "."
    )

    root = _init_tk()
    folder = filedialog.askdirectory(
        title="Select 'annotations' folder — contains *_annotations.json files",
        initialdir=default,
    )
    root.destroy()

    if not folder:
        print("❌  No folder selected. Exiting.")
        return None

    folder_path = Path(folder)
    json_files = list(folder_path.glob("*_annotations.json"))
    if not json_files:
        root2 = _init_tk()
        retry = messagebox.askyesno(
            "No annotation files found",
            f"No *_annotations.json files were found in:\n{folder}\n\n"
            "Do you want to select a different folder?",
        )
        root2.destroy()
        if retry:
            return _select_annotations_folder()
        print("⚠️   Proceeding with the selected folder.")

    print(f"✅  Annotations folder : {folder}")
    if json_files:
        print(f"    Found {len(json_files)} annotation file(s).")
    return folder


def _select_frames_folder():
    """Ask the user to select the frames (images) input folder."""
    print("\n" + "=" * 60)
    print("📁  STEP 2: Select FRAMES folder  (images)")
    print("=" * 60)
    print("Please choose the folder that contains the extracted frame images.")

    default = (
        str(Path("datasets/frames").resolve())
        if Path("datasets/frames").exists()
        else "."
    )

    root = _init_tk()
    folder = filedialog.askdirectory(
        title="Select 'frames' folder — contains extracted JPG/PNG images",
        initialdir=default,
    )
    root.destroy()

    if not folder:
        print("❌  No folder selected. Exiting.")
        return None

    folder_path = Path(folder)
    images = list(folder_path.rglob("*.jpg")) + list(folder_path.rglob("*.png"))
    if not images:
        root2 = _init_tk()
        retry = messagebox.askyesno(
            "No images found",
            f"No JPG/PNG images were found in:\n{folder}\n\n"
            "Do you want to select a different folder?",
        )
        root2.destroy()
        if retry:
            return _select_frames_folder()
        print("⚠️   Proceeding with the selected folder.")

    print(f"✅  Frames folder      : {folder}")
    if images:
        print(f"    Found {len(images)} image file(s).")
    return folder


def _select_output_folder():
    """Ask the user to select or confirm the yolo_data output folder."""
    print("\n" + "=" * 60)
    print("📁  STEP 3: Select OUTPUT folder  (yolo_data)")
    print("=" * 60)
    print("Please choose the folder where the YOLO dataset will be saved.")
    print("(It will be created if it does not exist.)")

    default = (
        str(Path("datasets/yolo_data").resolve())
        if Path("datasets/yolo_data").exists()
        else "."
    )

    root = _init_tk()
    folder = filedialog.askdirectory(
        title="Select 'yolo_data' output folder — YOLO dataset will be saved here",
        initialdir=default,
    )
    root.destroy()

    if not folder:
        folder = "datasets/yolo_data"
        print(f"⚠️   No folder selected. Using default: {folder}")
    else:
        print(f"✅  Output folder      : {folder}")

    return folder


if __name__ == "__main__":
    print("\n" + "🔬 " * 20)
    print("  SPERM DETECTION — YOLO Annotation Converter")
    print("🔬 " * 20)

    annotations_dir = _select_annotations_folder()
    if annotations_dir is None:
        raise SystemExit(1)

    images_dir = _select_frames_folder()
    if images_dir is None:
        raise SystemExit(1)

    output_dir = _select_output_folder()

    print("\n" + "=" * 60)
    print("📋  Configuration Summary")
    print("=" * 60)
    print(f"  📂 Annotations : {annotations_dir}")
    print(f"  📂 Frames      : {images_dir}")
    print(f"  📂 Output      : {output_dir}")
    print("=" * 60)

    root = _init_tk()
    proceed = messagebox.askyesno(
        "Confirm",
        f"Ready to convert annotations.\n\n"
        f"Annotations : {annotations_dir}\n"
        f"Frames      : {images_dir}\n"
        f"Output      : {output_dir}\n\n"
        "Proceed?",
    )
    root.destroy()

    if not proceed:
        print("❌  Cancelled by user.")
        raise SystemExit(0)

    print("\n🚀  Starting conversion...\n")
    converter = AnnotationConverter(
        annotations_dir,
        images_dir,
        output_dir,
        seed=42,
        keep_empty_labels=True,
    )
    converter.convert()