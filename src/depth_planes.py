"""Depth and plane estimation for perspective-aware compositing."""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN


class DepthPlaneEstimator:
    """Estimate depth and planar surfaces for perspective correction."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize depth and plane estimator.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        print("✓ Depth/plane estimator initialized (heuristic mode)")
    
    def estimate(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Estimate depth map and detect planar surfaces.
        
        Args:
            image: Input image (H, W, 3) BGR
        
        Returns:
            Dict with keys:
                - 'depth_map': Depth map (H, W) float, normalized [0, 1]
                - 'plane_masks': List of plane masks
                - 'plane_normals': List of plane normal vectors
                - 'plane_confidence': Confidence score per plane
        """
        H, W = image.shape[:2]
        
        # Heuristic depth estimation (simplified)
        depth_map = self._heuristic_depth(image)
        
        # Detect planar regions using RANSAC on depth
        plane_masks, plane_normals, plane_confidence = self._detect_planes(depth_map)
        
        return {
            'depth_map': depth_map,
            'plane_masks': plane_masks,
            'plane_normals': plane_normals,
            'plane_confidence': plane_confidence,
        }
    
    def _heuristic_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth using heuristic cues.
        
        Heuristics:
        - Vertical position (higher in image = farther)
        - Blur/focus (sharper = closer)
        - Color saturation (more saturated = closer)
        
        Args:
            image: Input image (H, W, 3) BGR
        
        Returns:
            Depth map (H, W) float [0, 1], where 0=near, 1=far
        """
        H, W = image.shape[:2]
        
        # Cue 1: Vertical position (top = far, bottom = near)
        y_coords = np.tile(np.linspace(0, 1, H)[:, None], (1, W))
        
        # Cue 2: Sharpness (blur indicates depth)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.abs(laplacian)
        sharpness = cv2.GaussianBlur(sharpness, (15, 15), 0)
        sharpness_norm = 1.0 - (sharpness - sharpness.min()) / (sharpness.max() - sharpness.min() + 1e-8)
        
        # Cue 3: Color saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        saturation_depth = 1.0 - saturation  # Less saturated = farther
        
        # Combine cues
        depth_map = (
            0.5 * y_coords +
            0.3 * sharpness_norm +
            0.2 * saturation_depth
        )
        
        # Normalize to [0, 1]
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        
        return depth_map.astype(np.float32)
    
    def _detect_planes(
        self,
        depth_map: np.ndarray,
        num_planes: int = 3
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Detect planar surfaces using simplified RANSAC on depth.
        
        Args:
            depth_map: Depth map (H, W) float
            num_planes: Maximum number of planes to detect
        
        Returns:
            (plane_masks, plane_normals, plane_confidence)
        """
        H, W = depth_map.shape
        
        # Downsample for speed
        scale = 0.25
        depth_small = cv2.resize(depth_map, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        H_s, W_s = depth_small.shape
        
        # Create coordinate grid
        y_coords, x_coords = np.mgrid[0:H_s, 0:W_s]
        
        # Stack coordinates with depth for clustering
        points = np.stack([
            x_coords.ravel(),
            y_coords.ravel(),
            depth_small.ravel() * 100  # Scale depth for clustering
        ], axis=1)
        
        # Use DBSCAN to find clusters (proxy for planes)
        clustering = DBSCAN(eps=10, min_samples=50).fit(points)
        labels = clustering.labels_
        
        # Extract plane masks
        plane_masks = []
        plane_normals = []
        plane_confidence = []
        
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in sorted(unique_labels)[:num_planes]:
            mask_small = (labels == label).reshape(H_s, W_s).astype(np.uint8)
            
            # Resize mask back to original size
            mask = cv2.resize(mask_small, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # Compute plane normal (simplified - just use average gradient)
            if mask.sum() > 100:
                region_depth = depth_map[mask > 0]
                depth_std = region_depth.std()
                
                # Confidence based on depth uniformity
                confidence = 1.0 / (1.0 + depth_std * 10)
                
                # Normal vector (simplified - assume mostly fronto-parallel)
                normal = np.array([0, 0, 1], dtype=np.float32)
                
                plane_masks.append(mask)
                plane_normals.append(normal)
                plane_confidence.append(confidence)
        
        return plane_masks, plane_normals, plane_confidence
    
    def compute_homography(
        self,
        plane_mask: np.ndarray,
        target_rect: Tuple[int, int, int, int]
    ) -> Optional[np.ndarray]:
        """
        Compute homography for plane-based perspective correction.
        
        Args:
            plane_mask: Binary mask of planar region
            target_rect: Target rectangle (x, y, w, h)
        
        Returns:
            Homography matrix (3x3) or None if not enough points
        """
        # Find corners of plane mask
        contours, _ = cv2.findContours(plane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) < 4:
            return None
        
        # Use first 4 points as source
        src_pts = approx[:4].reshape(4, 2).astype(np.float32)
        
        # Target rectangle corners
        x, y, w, h = target_rect
        dst_pts = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32)
        
        # Compute homography
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        
        return H


def test_depth_planes():
    """Test depth and plane estimation."""
    # Create test image with depth cues
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    # Add foreground (bottom, sharp, saturated)
    cv2.rectangle(img, (100, 400), (300, 550), (0, 0, 255), -1)
    
    # Add mid-ground (middle, medium blur)
    mid = np.ones((100, 200, 3), dtype=np.uint8) * 150
    mid = cv2.GaussianBlur(mid, (5, 5), 0)
    img[200:300, 300:500] = mid
    
    # Add background (top, blurred, desaturated)
    bg = np.ones((150, 700, 3), dtype=np.uint8)
    bg[:, :] = [180, 180, 200]
    bg = cv2.GaussianBlur(bg, (15, 15), 0)
    img[0:150, 50:750] = bg
    
    estimator = DepthPlaneEstimator()
    result = estimator.estimate(img)
    
    print(f"Depth map shape: {result['depth_map'].shape}")
    print(f"Depth range: [{result['depth_map'].min():.3f}, {result['depth_map'].max():.3f}]")
    print(f"Detected {len(result['plane_masks'])} planes")
    
    for i, conf in enumerate(result['plane_confidence']):
        print(f"  Plane {i+1}: confidence={conf:.3f}")
    
    return result


if __name__ == "__main__":
    test_depth_planes()
