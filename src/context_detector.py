"""
Context-aware surface detection for natural ad placement.

Detects surfaces where ads would naturally appear in real life:
- TV/monitor screens
- Wall poster areas
- Billboard spaces
- Bus/vehicle sides
- Store windows
- Picture frames
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class AdSurface:
    """A detected surface suitable for ad placement."""
    x: int
    y: int
    w: int
    h: int
    type: str  # 'screen', 'wall_poster', 'billboard', 'vehicle', 'frame', 'window'
    confidence: float
    corners: Optional[np.ndarray] = None  # For perspective correction
    

class ContextDetector:
    """Detect natural ad placement surfaces in images."""
    
    def __init__(self, verbose: bool = False):
        """
        Initialize context detector.
        
        Args:
            verbose: Whether to print detection info
        """
        self.verbose = verbose
        
        if self.verbose:
            print("✓ Context detector initialized")
    
    def detect_surfaces(self, image: np.ndarray) -> List[AdSurface]:
        """
        Detect all ad-friendly surfaces in the image.
        
        Args:
            image: Input image (H, W, 3) BGR
            
        Returns:
            List of detected AdSurface objects, sorted by confidence
        """
        H, W = image.shape[:2]
        surfaces = []
        
        # 1. Detect screens (TVs, monitors) - blue rectangular glowing regions
        surfaces.extend(self._detect_screens(image))
        
        # 2. Detect wall poster areas - rectangular regions on walls
        surfaces.extend(self._detect_wall_posters(image))
        
        # 3. Detect picture frames - rectangular frames on walls
        surfaces.extend(self._detect_frames(image))
        
        # 4. Detect billboards/signs - large rectangular outdoor surfaces
        surfaces.extend(self._detect_billboards(image))
        
        # 5. Detect vehicle sides - large flat surfaces (buses, trucks)
        surfaces.extend(self._detect_vehicles(image))
        
        # Sort by confidence
        surfaces.sort(key=lambda s: s.confidence, reverse=True)
        
        if self.verbose:
            print(f"  Detected {len(surfaces)} ad-friendly surfaces")
            for s in surfaces[:5]:
                print(f"    {s.type}: ({s.x}, {s.y}, {s.w}x{s.h}) conf={s.confidence:.2f}")
        
        return surfaces
    
    def _detect_screens(self, image: np.ndarray) -> List[AdSurface]:
        """
        Detect TV/monitor screens - typically rectangular with blue/bright glow.
        
        Strategy:
        1. Find bright, blue-ish regions
        2. Look for rectangular contours with aspect ratio 16:9 or 4:3
        3. Check for uniform color (screen content)
        """
        H, W = image.shape[:2]
        surfaces = []
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Detect blue screens (H: 100-130, S: 50-255, V: 100-255)
        lower_blue = np.array([100, 50, 100])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Also detect bright regions (any color, high value)
        _, _, v = cv2.split(hsv)
        bright_mask = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)[1]
        
        # Combine masks
        screen_mask = cv2.bitwise_or(blue_mask, bright_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_CLOSE, kernel)
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter by size (screens are usually 5-40% of image)
            if area < W * H * 0.05 or area > W * H * 0.4:
                continue
            
            # Check aspect ratio (typical screens: 16:9, 4:3, 21:9)
            aspect = w / h
            if not (1.2 < aspect < 2.5):
                continue
            
            # Check rectangularity (contour area vs bounding box area)
            contour_area = cv2.contourArea(cnt)
            rectangularity = contour_area / area if area > 0 else 0
            if rectangularity < 0.8:
                continue
            
            # Get corner points for perspective correction
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Calculate confidence based on features
            confidence = 0.7  # Base confidence for screens
            
            # Bonus for blue color (typical TV standby/screensaver)
            roi = image[y:y+h, x:x+w]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            blue_pixels = cv2.inRange(roi_hsv, lower_blue, upper_blue)
            blue_ratio = np.sum(blue_pixels > 0) / (w * h)
            if blue_ratio > 0.5:
                confidence += 0.2
            
            # Bonus for high rectangularity
            confidence += rectangularity * 0.1
            
            surfaces.append(AdSurface(
                x=x, y=y, w=w, h=h,
                type='screen',
                confidence=min(confidence, 1.0),
                corners=approx if len(approx) == 4 else None
            ))
        
        return surfaces
    
    def _detect_wall_posters(self, image: np.ndarray) -> List[AdSurface]:
        """
        Detect wall areas suitable for posters.
        
        Strategy:
        1. Find large uniform regions (walls)
        2. Look for rectangular empty areas on walls
        3. Check for vertical orientation and good lighting
        """
        H, W = image.shape[:2]
        surfaces = []
        
        # Convert to LAB for better uniformity detection
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Find uniform regions (low gradient)
        grad_x = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold to find uniform areas
        uniform_mask = (gradient_mag < 20).astype(np.uint8) * 255
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
        uniform_mask = cv2.morphologyEx(uniform_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of uniform regions
        contours, _ = cv2.findContours(uniform_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            
            # Filter by size (walls are usually large)
            if area < W * H * 0.15:
                continue
            
            # Check if it's in the upper-middle part of image (typical wall location)
            center_y = y + h / 2
            if center_y > H * 0.7:  # Skip floor regions
                continue
            
            # Create candidate poster areas within this uniform region
            # Typical poster sizes: portrait or landscape rectangles
            poster_sizes = [
                (w * 0.15, h * 0.25),  # Small portrait
                (w * 0.20, h * 0.30),  # Medium portrait
                (w * 0.25, h * 0.15),  # Small landscape
                (w * 0.30, h * 0.20),  # Medium landscape
            ]
            
            for pw, ph in poster_sizes:
                pw, ph = int(pw), int(ph)
                
                # Try different positions within the uniform region
                positions = [
                    (x + w * 0.1, y + h * 0.2),   # Upper left
                    (x + w * 0.5, y + h * 0.2),   # Upper center
                    (x + w * 0.7, y + h * 0.2),   # Upper right
                    (x + w * 0.3, y + h * 0.4),   # Middle left
                    (x + w * 0.6, y + h * 0.4),   # Middle right
                ]
                
                for px, py in positions:
                    px, py = int(px), int(py)
                    
                    # Check bounds
                    if px + pw > x + w or py + ph > y + h:
                        continue
                    
                    # Check if this area is also uniform (good for poster)
                    roi_uniform = uniform_mask[py:py+ph, px:px+pw]
                    uniformity = np.mean(roi_uniform) / 255.0
                    
                    if uniformity < 0.7:
                        continue
                    
                    confidence = 0.5 + uniformity * 0.3
                    
                    surfaces.append(AdSurface(
                        x=px, y=py, w=pw, h=ph,
                        type='wall_poster',
                        confidence=confidence
                    ))
        
        return surfaces
    
    def _detect_frames(self, image: np.ndarray) -> List[AdSurface]:
        """
        Detect picture frames on walls - rectangular bordered regions.
        
        Strategy:
        1. Find rectangular contours with borders
        2. Check for frame-like appearance (border + content)
        """
        H, W = image.shape[:2]
        surfaces = []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            # Approximate to polygon
            epsilon = 0.02 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Check if it's a rectangle (4 corners)
            if len(approx) != 4:
                continue
            
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            
            # Filter by size (frames are usually 2-15% of image)
            if area < W * H * 0.02 or area > W * H * 0.15:
                continue
            
            # Check aspect ratio (common frame ratios)
            aspect = w / h
            if not (0.7 < aspect < 1.5):  # Portrait or square frames
                continue
            
            # Check if on wall (upper part of image)
            center_y = y + h / 2
            if center_y > H * 0.65:
                continue
            
            confidence = 0.6
            
            surfaces.append(AdSurface(
                x=x, y=y, w=w, h=h,
                type='frame',
                confidence=confidence,
                corners=approx
            ))
        
        return surfaces
    
    def _detect_billboards(self, image: np.ndarray) -> List[AdSurface]:
        """
        Detect billboard/sign surfaces - large rectangular outdoor surfaces.
        """
        # For indoor scenes, skip billboard detection
        # This would be more relevant for outdoor images
        return []
    
    def _detect_vehicles(self, image: np.ndarray) -> List[AdSurface]:
        """
        Detect vehicle sides (buses, trucks) - large flat surfaces for ads.
        """
        # This would require object detection
        # Skip for now in heuristic mode
        return []
    
    def visualize_surfaces(self, image: np.ndarray, surfaces: List[AdSurface]) -> np.ndarray:
        """
        Visualize detected surfaces on image.
        
        Args:
            image: Input image
            surfaces: List of detected surfaces
            
        Returns:
            Image with surfaces drawn
        """
        vis = image.copy()
        
        # Color map for different surface types
        colors = {
            'screen': (0, 255, 255),      # Yellow
            'wall_poster': (255, 0, 255),  # Magenta
            'frame': (0, 255, 0),          # Green
            'billboard': (255, 0, 0),      # Blue
            'vehicle': (0, 165, 255),      # Orange
        }
        
        for i, surface in enumerate(surfaces[:10]):  # Show top 10
            color = colors.get(surface.type, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(vis, (surface.x, surface.y), 
                         (surface.x + surface.w, surface.y + surface.h), 
                         color, 2)
            
            # Draw label
            label = f"{surface.type} ({surface.confidence:.2f})"
            cv2.putText(vis, label, (surface.x, surface.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw corner points if available
            if surface.corners is not None:
                cv2.drawContours(vis, [surface.corners], -1, color, 3)
        
        return vis


def test_context_detector():
    """Test context detector on sample image."""
    import os
    
    # Find a test image
    test_paths = [
        "outputs/test_poster.jpg",
        "examples/poster1.jpg",
        "data/test.jpg"
    ]
    
    test_image = None
    for path in test_paths:
        if os.path.exists(path):
            test_image = cv2.imread(path)
            break
    
    if test_image is None:
        # Create a synthetic test image with a TV
        print("Creating synthetic test image with TV screen...")
        test_image = np.ones((600, 800, 3), dtype=np.uint8) * 200
        
        # Draw a TV screen (blue rectangle)
        cv2.rectangle(test_image, (50, 50), (400, 275), (200, 100, 0), -1)  # TV bezel
        cv2.rectangle(test_image, (60, 60), (390, 265), (255, 150, 0), -1)  # Blue screen
        
        # Draw some wall poster areas
        cv2.rectangle(test_image, (500, 100), (700, 300), (180, 180, 180), -1)  # Wall area
    
    detector = ContextDetector(verbose=True)
    
    print("\nDetecting ad-friendly surfaces...")
    surfaces = detector.detect_surfaces(test_image)
    
    print(f"\n✓ Found {len(surfaces)} surfaces")
    for i, s in enumerate(surfaces[:5], 1):
        print(f"  {i}. {s.type}: ({s.x}, {s.y}) {s.w}x{s.h} - confidence: {s.confidence:.2f}")
    
    # Visualize
    vis = detector.visualize_surfaces(test_image, surfaces)
    cv2.imwrite("outputs/detected_surfaces.jpg", vis)
    print(f"\n✓ Visualization saved to outputs/detected_surfaces.jpg")


if __name__ == "__main__":
    test_context_detector()
