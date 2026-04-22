import cv2
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple
import numpy as np

dataclass
class BBox:
    """Bounding box annotation"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    class_name: str = "sperm"
    
    def to_dict(self) -> dict:
        return asdict(self)

class AnnotationTool:
    def __init__(self, images_dir, annotations_dir="datasets/annotations"):
        self.images_dir = Path(images_dir)
        self.annotations_dir = Path(annotations_dir)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Get list of images
        self.image_files = sorted([
            f for f in self.images_dir.glob('*') 
            if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])
        
        self.current_idx = 0
        self.current_image = None
        self.current_bboxes: List[BBox] = []
        self.drawing = False
        self.start_point = None
        self.display_image = None
        
        print(f"📁 Found {len(self.image_files)} images to annotate")
        
    def load_annotations(self, image_path: Path) -> List[BBox]:
        """Load existing annotations for an image"""
        annotation_file = self.annotations_dir / f"{image_path.stem}_annotations.json"
        
        if annotation_file.exists():
            with open(annotation_file, 'r') as f:
                data = json.load(f)
                return [BBox(**bbox) for bbox in data['bboxes']]
        return []
    
    def save_annotations(self, image_path: Path):
        """Save current annotations to JSON"""
        annotation_file = self.annotations_dir / f"{image_path.stem}_annotations.json"
        
        data = {
            'image_name': image_path.name,
            'image_path': str(image_path),
            'num_bboxes': len(self.current_bboxes),
            'bboxes': [bbox.to_dict() for bbox in self.current_bboxes]
        }
        
        with open(annotation_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Saved {len(self.current_bboxes)} annotations for {image_path.name}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bboxes"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.display_image = self.current_image.copy()
                cv2.rectangle(self.display_image, self.start_point, (x, y), (0, 255, 0), 2)
                self.redraw_all_bboxes()
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                x_min = min(self.start_point[0], x)
                y_min = min(self.start_point[1], y)
                x_max = max(self.start_point[0], x)
                y_max = max(self.start_point[1], y)
                
                # Only add if bbox has reasonable size
                if (x_max - x_min) > 10 and (y_max - y_min) > 10:
                    bbox = BBox(x_min, y_min, x_max, y_max)
                    self.current_bboxes.append(bbox)
                    print(f"✏️  Added bbox #{len(self.current_bboxes)}")
                
                self.redraw_all_bboxes()
    
    def redraw_all_bboxes(self):
        """Redraw all bboxes on current image"""
        if self.display_image is None:
            self.display_image = self.current_image.copy()
        else:
            self.display_image = self.current_image.copy()
        
        for i, bbox in enumerate(self.current_bboxes):
            cv2.rectangle(self.display_image, 
                         (bbox.x_min, bbox.y_min), 
                         (bbox.x_max, bbox.y_max), 
                         (0, 255, 0), 2)
            cv2.putText(self.display_image, f"#{i+1}", 
                       (bbox.x_min, bbox.y_min - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def show_instructions(self):
        """Print usage instructions"""
        print("\n" + "="*60)
        print("🖱️  ANNOTATION TOOL - KEYBOARD SHORTCUTS")
        print("="*60)
        print("LEFT CLICK + DRAG   : Draw bounding box around sperm")
        print("U                   : Undo last annotation")
        print("S                   : Save annotations for current image")
        print("D                   : Delete all annotations (current image)")
        print("N or RIGHT ARROW    : Next image")
        print("P or LEFT ARROW     : Previous image")
        print("SPACE               : Jump to image number")
        print("Q or ESC            : Quit")
        print("="*60 + "\n")
    
    def annotate(self):
        """Main annotation loop"""
        self.show_instructions()
        
        while self.current_idx < len(self.image_files):
            image_path = self.image_files[self.current_idx]
            print(f"\n📸 Image {self.current_idx + 1}/{len(self.image_files)}: {image_path.name}")
            
            # Load image and annotations
            self.current_image = cv2.imread(str(image_path))
            if self.current_image is None:
                print(f"❌ Could not read image: {image_path}")
                self.current_idx += 1
                continue
            
            self.current_bboxes = self.load_annotations(image_path)
            self.display_image = self.current_image.copy()
            
            # Resize for display if too large
            height, width = self.current_image.shape[:2]
            if width > 1920 or height > 1080:
                scale = min(1920/width, 1080/height)
                self.display_image = cv2.resize(self.display_image, 
                                               (int(width*scale), int(height*scale)))
                self.current_image = self.display_image.copy()
            
            self.redraw_all_bboxes()
            
            # Setup window
            window_name = "Sperm Annotation Tool"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            print(f"Current annotations: {len(self.current_bboxes)}")
            
            while True:
                # Display image with info
                display = self.display_image.copy()
                text = f"Image {self.current_idx + 1}/{len(self.image_files)} | Bboxes: {len(self.current_bboxes)}"
                cv2.putText(display, text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # Q or ESC
                    self.save_annotations(image_path)
                    cv2.destroyAllWindows()
                    return
                
                elif key == ord('n') or key == 83:  # N or RIGHT ARROW
                    self.save_annotations(image_path)
                    self.current_idx += 1
                    break
                
                elif key == ord('p') or key == 81:  # P or LEFT ARROW
                    self.save_annotations(image_path)
                    self.current_idx = max(0, self.current_idx - 1)
                    break
                
                elif key == ord('u'):  # U - Undo
                    if self.current_bboxes:
                        self.current_bboxes.pop()
                        print(f"↩️  Undone. Remaining: {len(self.current_bboxes)}")
                        self.redraw_all_bboxes()
                
                elif key == ord('s'):  # S - Save
                    self.save_annotations(image_path)
                
                elif key == ord('d'):  # D - Delete all
                    self.current_bboxes = []
                    print("🗑️  All annotations cleared")
                    self.redraw_all_bboxes()
                
                elif key == ord(' '):  # SPACE - Jump to image
                    cv2.destroyAllWindows()
                    try:
                        img_num = int(input(f"Enter image number (1-{len(self.image_files)}): "))
                        self.current_idx = img_num - 1
                        if self.current_idx < 0 or self.current_idx >= len(self.image_files):
                            self.current_idx = 0
                        self.save_annotations(image_path)
                        break
                    except ValueError:
                        self.current_idx = self.current_idx
                        break

if __name__ == "__main__":
    IMAGES_DIR = "datasets/frames"      # Directory with extracted frames
    ANNOTATIONS_DIR = "datasets/annotations"  # Directory to save annotations
    
    tool = AnnotationTool(IMAGES_DIR, ANNOTATIONS_DIR)
    tool.annotate()