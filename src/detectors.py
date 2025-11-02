"""Protected content detection: faces, text, people, and objects."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from .config import (
    FACE_CONF_THRESHOLD,
    TEXT_CONF_THRESHOLD,
    OBJECT_CONF_THRESHOLD,
    PADDING_PX,
    PADDING_PCT,
)


class ProtectedContentDetector:
    """Detects and masks protected content (faces, text, people) in images."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize detectors.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self._init_detectors()
    
    def _init_detectors(self):
        """Initialize all detection models."""
        # Face detection - using OpenCV Haar Cascade as fallback
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_detector = cv2.CascadeClassifier(face_cascade_path)
        
        # Text detection - using OpenCV EAST or simple edge detection
        # For production, use: DB++, CRAFT, or PaddleOCR with PaddlePaddle
        self.text_detector = None  # Placeholder - will use EAST or edge-based
        
        # People segmentation - placeholder for now
        # For production: use DeepLabv3 or SegFormer from torchvision
        self.person_segmentor = None
        
        print("✓ Protected content detectors initialized (basic fallback mode)")
    
    def detect(self, image: np.ndarray, padding_px: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Detect all protected content and return combined mask.
        
        Args:
            image: Input image (H, W, 3) BGR
            padding_px: Padding to apply around detected regions (default from config)
        
        Returns:
            Dict with keys:
                - 'combined_mask': Binary mask (H, W) uint8, 255=protected
                - 'faces_mask': Face regions mask
                - 'text_mask': Text regions mask
                - 'people_mask': People regions mask
                - 'bboxes': List of (x, y, w, h, class_name, confidence)
        """
        H, W = image.shape[:2]
        padding = padding_px or max(PADDING_PX, int(min(H, W) * PADDING_PCT))
        
        # Initialize masks
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        faces_mask = np.zeros((H, W), dtype=np.uint8)
        text_mask = np.zeros((H, W), dtype=np.uint8)
        people_mask = np.zeros((H, W), dtype=np.uint8)
        
        bboxes = []
        
        # 1. Detect faces
        faces = self._detect_faces(image)
        for (x, y, w, h, conf) in faces:
            if conf >= FACE_CONF_THRESHOLD:
                faces_mask = self._add_bbox_to_mask(faces_mask, x, y, w, h, padding)
                bboxes.append((x, y, w, h, 'face', conf))
        
        # 2. Detect text regions
        text_boxes = self._detect_text(image)
        for (x, y, w, h, conf) in text_boxes:
            if conf >= TEXT_CONF_THRESHOLD:
                text_mask = self._add_bbox_to_mask(text_mask, x, y, w, h, padding)
                bboxes.append((x, y, w, h, 'text', conf))
        
        # 3. Detect people (optional, resource-intensive)
        # For now, skip or use simple background subtraction
        
        # Combine all masks
        combined_mask = cv2.bitwise_or(combined_mask, faces_mask)
        combined_mask = cv2.bitwise_or(combined_mask, text_mask)
        combined_mask = cv2.bitwise_or(combined_mask, people_mask)
        
        return {
            'combined_mask': combined_mask,
            'faces_mask': faces_mask,
            'text_mask': text_mask,
            'people_mask': people_mask,
            'bboxes': bboxes,
        }
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using Haar Cascade (basic fallback).
        
        Returns:
            List of (x, y, w, h, confidence)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Haar doesn't provide confidence, use fixed value
        return [(x, y, w, h, 0.9) for (x, y, w, h) in faces]
    
    def _detect_text(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect text regions using edge-based heuristic (basic fallback).
        For production: use EAST, DB++, or PaddleOCR.
        
        Returns:
            List of (x, y, w, h, confidence)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter by aspect ratio and size (heuristic for text)
            aspect_ratio = w / (h + 1e-5)
            area = w * h
            if 2 < aspect_ratio < 15 and 500 < area < 50000:
                text_boxes.append((x, y, w, h, 0.7))  # Fixed confidence
        
        return text_boxes
    
    def _add_bbox_to_mask(
        self,
        mask: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        padding: int
    ) -> np.ndarray:
        """
        Add bounding box to mask with padding.
        
        Args:
            mask: Binary mask to update
            x, y, w, h: Bounding box coordinates
            padding: Padding in pixels
        
        Returns:
            Updated mask
        """
        H, W = mask.shape
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(W, x + w + padding)
        y2 = min(H, y + h + padding)
        
        mask[y1:y2, x1:x2] = 255
        return mask


def test_detector():
    """Test the detector on a sample image."""
    # Create a test image with simple shapes
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw some rectangles to simulate faces/objects
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.rectangle(img, (300, 150), (500, 250), (0, 255, 0), -1)
    
    # Add some text-like patterns
    cv2.putText(img, "SAMPLE TEXT", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    detector = ProtectedContentDetector()
    result = detector.detect(img)
    
    print(f"Detected {len(result['bboxes'])} protected regions")
    print(f"Combined mask coverage: {(result['combined_mask'] > 0).sum() / result['combined_mask'].size * 100:.1f}%")
    
    return result


if __name__ == "__main__":
    test_detector()
