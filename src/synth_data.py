"""Synthetic data generator for training ad placement models."""

import cv2
import numpy as np
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont

from .saliency import SaliencyEstimator
from .config import MIN_AD_AREA_PCT, MAX_AD_AREA_PCT, ASPECT_RATIOS


class SyntheticDataGenerator:
    """Generate synthetic poster images with ground truth labels."""
    
    def __init__(self, output_dir: str = "data/synth"):
        """
        Initialize synthetic data generator.
        
        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.saliency_estimator = SaliencyEstimator()
        
        print(f"✓ Synthetic data generator initialized")
        print(f"  Output directory: {self.output_dir}")
    
    def generate_dataset(self, num_images: int = 100, size: Tuple[int, int] = (600, 800)):
        """
        Generate synthetic dataset.
        
        Args:
            num_images: Number of images to generate
            size: Image size (H, W)
        """
        print(f"\nGenerating {num_images} synthetic posters...")
        
        metadata = []
        
        for i in range(num_images):
            # Generate synthetic poster
            poster, protected_masks, protected_info = self._generate_poster(size)
            
            # Compute saliency
            saliency_map = self.saliency_estimator.estimate(poster)
            
            # Find safe regions
            safe_regions = self._find_safe_regions(
                poster.shape[:2],
                protected_masks,
                saliency_map
            )
            
            # Save image
            img_path = self.output_dir / f"poster_{i:05d}.jpg"
            cv2.imwrite(str(img_path), poster)
            
            # Save metadata
            meta = {
                'image': img_path.name,
                'size': {'width': size[1], 'height': size[0]},
                'protected_regions': protected_info,
                'safe_regions': [
                    {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
                    for (x, y, w, h) in safe_regions
                ],
            }
            metadata.append(meta)
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{num_images} images")
        
        # Save metadata JSON
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Dataset generation complete")
        print(f"  Images saved to: {self.output_dir}")
        print(f"  Metadata saved to: {meta_path}")
    
    def _generate_poster(self, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Generate a synthetic poster with protected content.
        
        Returns:
            (poster_image, protected_mask, protected_info)
        """
        H, W = size
        
        # Random background color
        bg_color = np.random.randint(150, 230, size=3, dtype=np.uint8)
        poster = np.ones((H, W, 3), dtype=np.uint8) * bg_color
        
        # Add gradient
        if random.random() < 0.5:
            gradient = np.linspace(0, 1, H)[:, None]
            gradient = np.tile(gradient, (1, W))[:, :, None]
            poster = (poster * (0.7 + 0.3 * gradient)).astype(np.uint8)
        
        # Protected mask
        protected_mask = np.zeros((H, W), dtype=np.uint8)
        protected_info = []
        
        # Add text regions (1-3)
        num_text = random.randint(1, 3)
        for _ in range(num_text):
            text_mask, text_bbox = self._add_text_region(poster, H, W)
            protected_mask = np.maximum(protected_mask, text_mask)
            protected_info.append({
                'type': 'text',
                'bbox': text_bbox,
            })
        
        # Add face-like ellipses (0-2)
        num_faces = random.randint(0, 2)
        for _ in range(num_faces):
            face_mask, face_bbox = self._add_face_region(poster, H, W)
            protected_mask = np.maximum(protected_mask, face_mask)
            protected_info.append({
                'type': 'face',
                'bbox': face_bbox,
            })
        
        # Add logo-like shapes (0-2)
        num_logos = random.randint(0, 2)
        for _ in range(num_logos):
            logo_mask, logo_bbox = self._add_logo_region(poster, H, W)
            protected_mask = np.maximum(protected_mask, logo_mask)
            protected_info.append({
                'type': 'logo',
                'bbox': logo_bbox,
            })
        
        return poster, protected_mask, protected_info
    
    def _add_text_region(self, poster: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, Dict]:
        """Add text region to poster."""
        # Random text
        words = ["SALE", "EVENT", "CONCERT", "SPECIAL", "LIMITED", "OFFER", "NEW"]
        text = random.choice(words)
        
        # Random position and size
        font_size = random.randint(30, 80)
        x = random.randint(50, W - 200)
        y = random.randint(50, H - 100)
        
        # Draw text
        color = tuple(map(int, np.random.randint(0, 100, 3)))
        cv2.putText(poster, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_size / 30, color, max(2, font_size // 20))
        
        # Approximate bbox
        text_w = len(text) * font_size // 2
        text_h = font_size
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask, (x - 10, y - text_h - 10), 
                     (x + text_w + 10, y + 10), 255, -1)
        
        bbox = {'x': x - 10, 'y': y - text_h - 10, 
                'w': text_w + 20, 'h': text_h + 20}
        
        return mask, bbox
    
    def _add_face_region(self, poster: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, Dict]:
        """Add face-like ellipse to poster."""
        # Random position and size
        cx = random.randint(100, W - 100)
        cy = random.randint(100, H - 100)
        w = random.randint(60, 120)
        h = int(w * 1.3)
        
        # Draw ellipse
        color = tuple(map(int, np.random.randint(180, 220, 3)))
        cv2.ellipse(poster, (cx, cy), (w // 2, h // 2), 0, 0, 360, color, -1)
        
        # Add eyes and mouth (simple)
        cv2.circle(poster, (cx - w // 4, cy - h // 6), w // 10, (50, 50, 50), -1)
        cv2.circle(poster, (cx + w // 4, cy - h // 6), w // 10, (50, 50, 50), -1)
        cv2.ellipse(poster, (cx, cy + h // 6), (w // 4, h // 8), 0, 0, 180, (50, 50, 50), 2)
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.ellipse(mask, (cx, cy), (w // 2 + 20, h // 2 + 20), 0, 0, 360, 255, -1)
        
        bbox = {'x': cx - w // 2 - 20, 'y': cy - h // 2 - 20,
                'w': w + 40, 'h': h + 40}
        
        return mask, bbox
    
    def _add_logo_region(self, poster: np.ndarray, H: int, W: int) -> Tuple[np.ndarray, Dict]:
        """Add logo-like geometric shape to poster."""
        # Random position and size
        x = random.randint(50, W - 150)
        y = random.randint(50, H - 150)
        size = random.randint(40, 100)
        
        # Draw shape
        color = tuple(map(int, np.random.randint(100, 200, 3)))
        shape = random.choice(['circle', 'square', 'triangle'])
        
        if shape == 'circle':
            cv2.circle(poster, (x + size // 2, y + size // 2), size // 2, color, -1)
        elif shape == 'square':
            cv2.rectangle(poster, (x, y), (x + size, y + size), color, -1)
        else:  # triangle
            pts = np.array([[x + size // 2, y], 
                           [x, y + size], 
                           [x + size, y + size]], np.int32)
            cv2.fillPoly(poster, [pts], color)
        
        # Create mask
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.rectangle(mask, (x - 10, y - 10), (x + size + 10, y + size + 10), 255, -1)
        
        bbox = {'x': x - 10, 'y': y - 10, 'w': size + 20, 'h': size + 20}
        
        return mask, bbox
    
    def _find_safe_regions(
        self,
        image_shape: Tuple[int, int],
        protected_mask: np.ndarray,
        saliency_map: np.ndarray,
        num_regions: int = 10
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find safe regions that don't overlap protected content.
        
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        H, W = image_shape
        total_area = H * W
        min_area = int(total_area * MIN_AD_AREA_PCT)
        max_area = int(total_area * MAX_AD_AREA_PCT)
        
        safe_regions = []
        max_attempts = 500
        
        for _ in range(max_attempts):
            if len(safe_regions) >= num_regions:
                break
            
            # Random aspect ratio and size
            ar_w, ar_h = random.choice(ASPECT_RATIOS)
            area = random.uniform(min_area, max_area)
            w = int((area * ar_w / ar_h) ** 0.5)
            h = int(area / w) if w > 0 else 0
            
            if w <= 0 or h <= 0 or w > W or h > H:
                continue
            
            # Random position
            x = random.randint(0, max(1, W - w))
            y = random.randint(0, max(1, H - h))
            
            # Check if safe
            region_protected = protected_mask[y:y+h, x:x+w]
            region_saliency = saliency_map[y:y+h, x:x+w]
            
            if not region_protected.any() and region_saliency.mean() < 0.5:
                safe_regions.append((x, y, w, h))
        
        return safe_regions


def demo_synthetic_generator():
    """Demo the synthetic data generator."""
    generator = SyntheticDataGenerator(output_dir="data/synth_demo")
    
    # Generate small test dataset
    generator.generate_dataset(num_images=10, size=(600, 800))
    
    print("\n✓ Demo complete. Check data/synth_demo/ for generated images")


if __name__ == "__main__":
    demo_synthetic_generator()
