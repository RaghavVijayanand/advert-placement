"""
Advanced edge detection for precise TV screen boundary detection.
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import time


class PreciseEdgeDetector:
    """Detect precise edges of TV screens using multi-scale edge detection."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def detect_screen_edges(self, image: np.ndarray, initial_bbox: Tuple[int, int, int, int],
                           expand_margin: int = 40) -> Tuple[int, int, int, int]:
        """
        Refine screen bounding box using contour-based edge detection.
        
        Args:
            image: Input image
            initial_bbox: Initial (x, y, w, h) from context detector
            expand_margin: Pixels to expand search region
        
        Returns:
            Refined (x, y, w, h) bounding box
        """
        x, y, w, h = initial_bbox
        H, W = image.shape[:2]
        
        if self.verbose:
            print(f"  Refining edges for screen at ({x}, {y}) {w}x{h}")
        
        t0 = time.time()
        
        # Expand search region
        x1 = max(0, x - expand_margin)
        y1 = max(0, y - expand_margin)
        x2 = min(W, x + w + expand_margin)
        y2 = min(H, y + h + expand_margin)
        
        # Extract region of interest
        roi = image[y1:y2, x1:x2].copy()
        roi_h, roi_w = roi.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Use screen color detection to find precise boundaries
        screen_mask = self._detect_screen_mask(roi)
        
        # Edge detection
        edges = self._detect_edges(gray)
        
        # Save debug images
        cv2.imwrite('outputs/debug_edges.jpg', edges)
        cv2.imwrite('outputs/debug_screen_mask_edges.jpg', screen_mask)
        
        # Combine edges with screen mask to find screen boundaries
        screen_edges = cv2.bitwise_and(edges, screen_mask)
        
        # Find contours in the screen mask (not edges)
        contours, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if self.verbose:
                print(f"  ✗ No contours found, using initial bbox")
            return initial_bbox
        
        # Find the largest contour (likely the screen)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        rect_x, rect_y, rect_w, rect_h = cv2.boundingRect(largest_contour)
        
        # Check if it's reasonable
        area = cv2.contourArea(largest_contour)
        rect_area = rect_w * rect_h
        
        if rect_area == 0 or area < w * h * 0.3:
            if self.verbose:
                print(f"  ✗ Contour too small, using initial bbox")
            return initial_bbox
        
        rectangularity = area / rect_area
        
        if rectangularity < 0.6:
            if self.verbose:
                print(f"  ✗ Low rectangularity ({rectangularity:.2f}), using initial bbox")
            return initial_bbox
        
        # Convert back to full image coordinates
        new_x = x1 + rect_x
        new_y = y1 + rect_y
        new_w = rect_w
        new_h = rect_h
        
        # Sanity checks
        if new_w < 100 or new_h < 100:
            if self.verbose:
                print(f"  ✗ Invalid dimensions ({new_w}x{new_h}), using initial bbox")
            return initial_bbox
        
        # Make sure it's within image bounds
        new_x = max(0, min(new_x, W - 10))
        new_y = max(0, min(new_y, H - 10))
        new_w = min(new_w, W - new_x)
        new_h = min(new_h, H - new_y)
        
        if self.verbose:
            area_change = 100 * (new_w * new_h - w * h) / (w * h)
            print(f"  ✓ Edges refined in {time.time() - t0:.2f}s")
            print(f"    Initial: ({x}, {y}) {w}x{h}")
            print(f"    Refined: ({new_x}, {new_y}) {new_w}x{new_h}")
            print(f"    Area change: {area_change:+.1f}%")
            print(f"    Rectangularity: {rectangularity:.2f}")
        
        return (new_x, new_y, new_w, new_h)
    
    def _detect_screen_mask(self, roi: np.ndarray) -> np.ndarray:
        """Detect screen region in ROI using color."""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Detect blue screen pixels (TVs are often blue/bright)
        lower_blue = np.array([80, 20, 80])
        upper_blue = np.array([150, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Also detect bright pixels
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        bright_mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1]
        
        # Combine
        screen_mask = cv2.bitwise_or(blue_mask, bright_mask)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_OPEN, kernel)
        
        return screen_mask
    
    def _detect_edges(self, gray: np.ndarray) -> np.ndarray:
        """Apply edge detection with proper cleanup."""
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    

    def visualize_edges(self, image: np.ndarray, initial_bbox: Tuple[int, int, int, int],
                       refined_bbox: Tuple[int, int, int, int], output_path: str):
        """Visualize edge detection results."""
        vis = image.copy()
        
        # Draw initial bbox (red)
        x, y, w, h = initial_bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(vis, "Initial", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw refined bbox (green)
        x, y, w, h = refined_bbox
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(vis, "Refined", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, vis)
        print(f"✓ Edge visualization saved to {output_path}")


def test_edge_detection():
    """Test edge detection on living room image."""
    print("=" * 60)
    print("Testing Precise Edge Detection")
    print("=" * 60)
    
    # Load test image
    image_path = 'examples/living_room.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")
    
    # Initial bbox from context detector
    initial_bbox = (49, 180, 420, 267)
    
    # Initialize edge detector
    detector = PreciseEdgeDetector(verbose=True)
    
    # Detect edges
    refined_bbox = detector.detect_screen_edges(image, initial_bbox, expand_margin=30)
    
    # Visualize
    detector.visualize_edges(image, initial_bbox, refined_bbox, 'outputs/edge_refinement.jpg')
    
    print("\n" + "="*60)
    print("Edge detection test complete!")


if __name__ == "__main__":
    test_edge_detection()
