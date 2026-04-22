import json
import os
from pathlib import Path
from typing import List, Tuple
import random
import shutil
from dataclasses import dataclass
import yaml

@dataclass
class YOLODataset:
    image_path: str
    label_path: str
    class_id: int = 0
    
class AnnotationConverter:
    def __init__(self, annotations_dir, images_dir, output_dir="datasets/yolo_data"):
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.class_names = ["sperm"]
        
        # Create output directory structure
        self.setup_directories()
        
    def setup_directories(self):
        """Create YOLO dataset directory structure"""
        splits = ["train", "val", "test"]
        subdirs = ["images", "labels"]
        
        for split in splits:
            for subdir in subdirs:
                path = self.output_dir / split / subdir
                path.mkdir(parents=True, exist_ok=True)
    
    def convert_bbox_to_yolo(self, bbox: dict, image_width: int, image_height: int) -> str:
        """
        Convert bbox from pixel coordinates to YOLO format
        YOLO format: <class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
        """
        x_min = bbox['x_min']
        y_min = bbox['y_min']
        x_max = bbox['x_max']
        y_max = bbox['y_max']
        
        # Calculate center and dimensions
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        
        # Clamp values between 0 and 1
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        return f"0 {{x_center:.6f}} {{y_center:.6f}} {{width:.6f}} {{height:.6f}}\n"
    
    def read_image_size(self, image_path: Path) -> Tuple[int, int]:
        """Read image dimensions"""
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        height, width = img.shape[:2]
        return width, height
    
    def convert_annotations(self) -> List[Tuple[Path, Path]]:
        """
        Convert all annotations to YOLO format
        Returns list of (image_path, label_path) tuples
        """
        annotation_files = list(self.annotations_dir.glob('*_annotations.json'))
        dataset_pairs = []
        
        print(f"🔄 Converting {len(annotation_files)} annotation files...")
        
        for annotation_file in annotation_files:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Find corresponding image
            image_name = data['image_name']
            image_path = self.images_dir / image_name
            
            if not image_path.exists():
                print(f"⚠️  Image not found: {image_path}")
                continue
            
            # Get image dimensions
            img_size = self.read_image_size(image_path)
            if img_size is None:
                print(f"⚠️  Could not read image: {image_path}")
                continue
            
            image_width, image_height = img_size
            
            # Convert bboxes to YOLO format
            yolo_labels = []
            for bbox in data['bboxes']:
                yolo_line = self.convert_bbox_to_yolo(bbox, image_width, image_height)
                yolo_labels.append(yolo_line)
            
            # Create label file
            label_filename = f"{image_path.stem}.txt"
            label_path = self.output_dir / "temp" / label_filename
            label_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(label_path, 'w') as f:
                f.writelines(yolo_labels)
            
            dataset_pairs.append((image_path, label_path))
            
            if len(dataset_pairs) % 100 == 0:
                print(f"✅ Converted {len(dataset_pairs)} images")
        
        print(f"✅ Conversion complete: {len(dataset_pairs)} images")
        return dataset_pairs
    
    def split_dataset(self, dataset_pairs: List[Tuple[Path, Path]], 
                     train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """Split dataset into train/val/test and copy files"""
        
        # Shuffle dataset
        random.shuffle(dataset_pairs)
        
        total = len(dataset_pairs)
        train_idx = int(total * train_ratio)
        val_idx = int(total * (train_ratio + val_ratio))
        
        splits = {
            'train': dataset_pairs[:train_idx],
            'val': dataset_pairs[train_idx:val_idx],
            'test': dataset_pairs[val_idx:]
        }
        
        print(f"\n📊 Dataset split:")
        print(f"  Train: {len(splits['train'])} images ({train_ratio*100:.0f}%)")
        print(f"  Val:   {len(splits['val'])} images ({val_ratio*100:.0f}%)")
        print(f"  Test:  {len(splits['test'])} images ({test_ratio*100:.0f}%)")
        
        # Copy files to respective directories
        for split_name, pairs in splits.items():
            print(f"\n📁 Copying {split_name} images...")
            for img_path, label_path in pairs:
                # Copy image
                dest_img = self.output_dir / split_name / "images" / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Copy label
                dest_label = self.output_dir / split_name / "labels" / label_path.name
                shutil.copy2(label_path, dest_label)
            
            print(f"✅ {split_name} split copied")
        
        # Cleanup temp directory
        temp_dir = self.output_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    
    def create_dataset_yaml(self):
        """Create dataset.yaml for YOLO training"""
        dataset_yaml = {
            'path': str(self.output_dir.absolute()),
            'train': str((self.output_dir / 'train' / 'images').relative_to(self.output_dir)),
            'val': str((self.output_dir / 'val' / 'images').relative_to(self.output_dir)),
            'test': str((self.output_dir / 'test' / 'images').relative_to(self.output_dir)),
            'nc': 1,
            'names': {0: 'sperm'}
        }
        
        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(dataset_yaml, f, default_flow_style=False)
        
        print(f"\n✅ Created dataset.yaml at: {yaml_path}")
        return yaml_path
    
    def generate_statistics(self):
        """Generate dataset statistics"""
        print("\n" + "="*60)
        print("📊 DATASET STATISTICS")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            labels_dir = self.output_dir / split / 'labels'
            images_dir = self.output_dir / split / 'images'
            
            label_files = list(labels_dir.glob('*.txt'))
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            
            total_objects = 0
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    total_objects += len(f.readlines())
            
            avg_objects = total_objects / len(label_files) if label_files else 0
            
            print(f"\n{split.upper()}:")
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            print(f"  Total objects: {total_objects}")
            print(f"  Avg objects per image: {avg_objects:.2f}")
    
    def convert(self):
        """Main conversion pipeline"""
        print("🚀 Starting YOLO annotation conversion...\n")
        
        # Convert annotations
        dataset_pairs = self.convert_annotations()
        
        if not dataset_pairs:
            print("❌ No valid dataset pairs found!")
            return
        
        # Split dataset
        self.split_dataset(dataset_pairs)
        
        # Create dataset.yaml
        self.create_dataset_yaml()
        
        # Generate statistics
        self.generate_statistics()
        
        print(f"\n✅ YOLO dataset ready at: {self.output_dir}")

if __name__ == "__main__":
    ANNOTATIONS_DIR = "datasets/annotations"
    IMAGES_DIR = "datasets/frames"
    OUTPUT_DIR = "datasets/yolo_data"
    
    converter = AnnotationConverter(ANNOTATIONS_DIR, IMAGES_DIR, OUTPUT_DIR)
    converter.convert()