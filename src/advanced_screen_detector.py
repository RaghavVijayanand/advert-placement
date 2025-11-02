"""
Advanced screen detection and perspective-aware ad placement.

Uses edge detection, depth estimation, and perspective transforms to
accurately place ads on TV screens and other surfaces.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class ScreenSurface:
    """Detected screen surface with 3D information."""
    corners: np.ndarray  # 4 corner points in order: TL, TR, BR, BL
    confidence: float
    depth_map: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None


class AdvancedScreenDetector:
    """Advanced screen detection using edge detection and depth."""
    
    def __init__(self, verbose: bool = False):
        """Initialize detector."""
        self.verbose = verbose
        
        if self.verbose:
            print("✓ Advanced screen detector initialized")
            print("  Using: Canny edges + Hough lines + depth estimation")
    
    def detect_screen(self, image: np.ndarray) -> Optional[ScreenSurface]:
        """
        Detect TV/monitor screen with precise boundaries.
        
        Args:
            image: Input image (H, W, 3) BGR
            
        Returns:
            ScreenSurface with corners and mask, or None if not found
        """
        H, W = image.shape[:2]
        
        if self.verbose:
            print(f"\nDetecting screen in {W}x{H} image...")
        
        # Step 1: Find screen region using color and brightness
        screen_mask = self._detect_screen_region(image)
        cv2.imwrite('outputs/debug_screen_mask.jpg', screen_mask)
        
        # Step 2: Refine with edge detection
        edges = self._detect_edges(image)
        cv2.imwrite('outputs/debug_edges.jpg', edges)
        
        # Step 3: Find rectangular contours
        screen_corners = self._find_screen_corners(edges, screen_mask, (H, W))
        
        if screen_corners is None:
            if self.verbose:
                print("  ✗ Could not find screen corners")
            return None
        
        # Step 4: Create precise mask
        screen_exact_mask = self._create_screen_mask(screen_corners, (H, W))
        
        # Step 5: Estimate depth
        depth_map = self._estimate_depth(image)
        
        # Step 6: Refine mask using depth (screen is further back than people)
        screen_final_mask = self._refine_with_depth(screen_exact_mask, depth_map, image)
        
        confidence = self._calculate_confidence(screen_corners, screen_final_mask)
        
        if self.verbose:
            print(f"  ✓ Screen detected with {confidence:.1%} confidence")
            print(f"    Corners: {screen_corners.reshape(-1, 2).tolist()}")
        
        return ScreenSurface(
            corners=screen_corners,
            confidence=confidence,
            depth_map=depth_map,
            mask=screen_final_mask
        )
    
    def _detect_screen_region(self, image: np.ndarray) -> np.ndarray:
        """Detect screen region using color/brightness."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Screens are typically bright and blue-ish (relaxed thresholds)
        blue_mask = cv2.inRange(hsv, np.array([80, 20, 80]), np.array([150, 255, 255]))
        bright_mask = cv2.threshold(v, 120, 255, cv2.THRESH_BINARY)[1]
        
        # Also detect high saturation blue regions
        blue_sat = cv2.inRange(hsv, np.array([90, 100, 50]), np.array([130, 255, 255]))
        
        # Combine all masks
        screen_mask = cv2.bitwise_or(blue_mask, bright_mask)
        screen_mask = cv2.bitwise_or(screen_mask, blue_sat)
        
        # Morphology to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        screen_mask = cv2.morphologyEx(screen_mask, cv2.MORPH_OPEN, kernel)
        
        if self.verbose:
            screen_pixels = np.sum(screen_mask > 0)
            total_pixels = screen_mask.shape[0] * screen_mask.shape[1]
            print(f"  Screen region: {screen_pixels} pixels ({100*screen_pixels/total_pixels:.1f}%)")
        
        return screen_mask
    
    def _detect_edges(self, image: np.ndarray) -> np.ndarray:
        """Detect edges using Canny."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Dilate to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        return edges
    
    def _find_screen_corners(self, edges: np.ndarray, screen_mask: np.ndarray, 
                            image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """Find the 4 corners of the screen using edges and mask."""
        H, W = image_shape
        
        # Find contours in the screen mask
        contours, _ = cv2.findContours(screen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter contours by size and aspect ratio (screens are rectangular)
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Size filter: 3-40% of image
            if area < H * W * 0.03 or area > H * W * 0.4:
                continue
            
            # Aspect ratio filter
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            
            # Typical screen ratios: 16:9 (1.78), 4:3 (1.33), 21:9 (2.33)
            if not (1.2 < aspect < 2.5):
                continue
            
            # Rectangularity filter
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            rectangularity = area / hull_area if hull_area > 0 else 0
            if rectangularity < 0.75:
                continue
            
            valid_contours.append((cnt, area, aspect, rectangularity))
        
        if not valid_contours:
            if self.verbose:
                print(f"  No valid screen contours found (checked {len(contours)} total)")
            return None
        
        # Get best contour (highest rectangularity * area)
        best_contour = max(valid_contours, key=lambda x: x[3] * (x[1] / (H * W)))
        largest_contour = best_contour[0]
        area = best_contour[1]
        
        if self.verbose:
            print(f"  Selected contour: area={100*area/(H*W):.1f}%, aspect={best_contour[2]:.2f}, rect={best_contour[3]:.2f}")
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If not 4 corners, use bounding rect
        if len(approx) != 4:
            x, y, w, h = cv2.boundingRect(largest_contour)
            corners = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]]
            ], dtype=np.float32)
        else:
            corners = approx.astype(np.float32)
        
        # Order corners: TL, TR, BR, BL
        corners = self._order_corners(corners.reshape(-1, 2))
        
        return corners
    
    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order points as TL, TR, BR, BL."""
        # Sort by y-coordinate
        sorted_by_y = pts[np.argsort(pts[:, 1])]
        
        # Top two points
        top = sorted_by_y[:2]
        top = top[np.argsort(top[:, 0])]  # Sort by x
        tl, tr = top[0], top[1]
        
        # Bottom two points
        bottom = sorted_by_y[2:]
        bottom = bottom[np.argsort(bottom[:, 0])]
        bl, br = bottom[0], bottom[1]
        
        return np.array([[tl], [tr], [br], [bl]], dtype=np.float32)
    
    def _create_screen_mask(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """Create precise mask from corners."""
        H, W = image_shape
        mask = np.zeros((H, W), dtype=np.uint8)
        
        corners_int = corners.reshape(-1, 2).astype(np.int32)
        cv2.fillPoly(mask, [corners_int], 255)
        
        return mask
    
    def _estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map using heuristics.
        For better results, use MiDaS or other depth estimation models.
        """
        H, W = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Heuristic depth based on:
        # 1. Vertical position (higher = further)
        y_coords = np.arange(H).reshape(-1, 1) / H
        y_depth = np.tile(y_coords, (1, W))
        
        # 2. Brightness (brighter = closer, typically)
        brightness = gray.astype(np.float32) / 255.0
        brightness_depth = 1.0 - brightness
        
        # 3. Edge density (more edges = closer)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0) / 255.0
        edge_depth = 1.0 - edge_density
        
        # Combine
        depth = (
            0.4 * y_depth +
            0.3 * brightness_depth +
            0.3 * edge_depth
        )
        
        # Normalize
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        return depth.astype(np.float32)
    
    def _refine_with_depth(self, screen_mask: np.ndarray, depth_map: np.ndarray, 
                          image: np.ndarray) -> np.ndarray:
        """
        Refine screen mask using depth information.
        Screen should be at consistent depth, people are closer.
        """
        # Get depth values within screen region
        screen_depth_values = depth_map[screen_mask > 0]
        
        if len(screen_depth_values) == 0:
            return screen_mask
        
        # Screen should be at far depth (high values)
        median_screen_depth = np.median(screen_depth_values)
        std_screen_depth = np.std(screen_depth_values)
        
        # Create depth-based mask (keep only far pixels)
        # Anything significantly closer than screen median is likely a person
        depth_threshold = median_screen_depth - 0.15  # Closer than this = person
        depth_mask = (depth_map >= depth_threshold).astype(np.uint8) * 255
        
        # Combine with original screen mask
        refined_mask = cv2.bitwise_and(screen_mask, depth_mask)
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel)
        refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
        
        return refined_mask
    
    def _calculate_confidence(self, corners: np.ndarray, mask: np.ndarray) -> float:
        """Calculate confidence score for detection."""
        # Check rectangularity
        contour_area = cv2.contourArea(corners)
        x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
        bbox_area = w * h
        
        if bbox_area == 0:
            return 0.0
        
        rectangularity = contour_area / bbox_area
        
        # Check mask fill
        mask_area = np.sum(mask > 0)
        fill_ratio = mask_area / bbox_area if bbox_area > 0 else 0
        
        # Combine
        confidence = 0.6 * rectangularity + 0.4 * fill_ratio
        
        return float(confidence)
    
    def apply_ad_to_screen(self, image: np.ndarray, ad: np.ndarray, 
                          surface: ScreenSurface) -> np.ndarray:
        """
        Apply ad to screen surface with perspective transform.
        
        Args:
            image: Original image
            ad: Ad image to place
            surface: Detected screen surface
            
        Returns:
            Image with ad composited onto screen
        """
        H, W = image.shape[:2]
        ad_h, ad_w = ad.shape[:2]
        
        # Define destination points (ad corners)
        dst_pts = np.array([
            [0, 0],
            [ad_w, 0],
            [ad_w, ad_h],
            [0, ad_h]
        ], dtype=np.float32)
        
        # Source points (screen corners in image)
        src_pts = surface.corners.reshape(-1, 2).astype(np.float32)
        
        # Compute homography (perspective transform from ad to screen)
        H_matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        
        # Warp ad to screen perspective
        ad_warped = cv2.warpPerspective(ad, H_matrix, (W, H))
        
        # Create alpha mask for the warped ad (only within screen bounds)
        ad_mask = surface.mask.copy()
        
        # Smooth mask edges
        ad_mask = cv2.GaussianBlur(ad_mask, (7, 7), 0)
        ad_mask = ad_mask.astype(np.float32) / 255.0
        ad_mask = np.expand_dims(ad_mask, axis=2)
        
        # Blend: ad where mask is white, original image elsewhere
        result = (
            ad_mask * ad_warped.astype(np.float32) +
            (1.0 - ad_mask) * image.astype(np.float32)
        ).astype(np.uint8)
        
        return result


def test_advanced_detector():
    """Test advanced screen detection on living room image."""
    import os
    
    # Load image
    image_path = 'image.png'
    if not os.path.exists(image_path):
        image_path = 'examples/living_room.jpg'
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")
    
    # Detect screen
    detector = AdvancedScreenDetector(verbose=True)
    surface = detector.detect_screen(image)
    
    if surface is None:
        print("\n✗ No screen detected")
        return
    
    print(f"\n✓ Screen detected successfully!")
    print(f"  Confidence: {surface.confidence:.1%}")
    print(f"  Corners: {surface.corners.reshape(-1, 2).tolist()}")
    
    # Visualize detection
    vis = image.copy()
    
    # Draw corners
    corners_int = surface.corners.reshape(-1, 2).astype(np.int32)
    cv2.polylines(vis, [corners_int], True, (0, 255, 0), 3)
    
    # Draw corner points
    for i, (x, y) in enumerate(corners_int):
        cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
        cv2.putText(vis, str(i), (int(x)+15, int(y)+15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save visualization
    cv2.imwrite('outputs/screen_detection_advanced.jpg', vis)
    print("\n✓ Detection visualization saved to outputs/screen_detection_advanced.jpg")
    
    # Save mask
    cv2.imwrite('outputs/screen_mask_refined.jpg', surface.mask)
    print("✓ Refined mask saved to outputs/screen_mask_refined.jpg")
    
    # Save depth map
    depth_vis = (surface.depth_map * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    cv2.imwrite('outputs/depth_map.jpg', depth_colored)
    print("✓ Depth map saved to outputs/depth_map.jpg")
    
    return surface


if __name__ == "__main__":
    test_advanced_detector()
