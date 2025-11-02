"""Semantic segmentation for uniform background detection."""

import cv2
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans


class SemanticSegmenter:
    """Segment image into semantic regions (sky, wall, floor, etc.)."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize semantic segmenter.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = device
        print("✓ Semantic segmenter initialized (heuristic mode)")
    
    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment image into semantic regions.
        
        Args:
            image: Input image (H, W, 3) BGR
        
        Returns:
            Dict with keys:
                - 'segmentation_map': Semantic labels (H, W) uint8
                - 'class_masks': Dict of class_name -> binary mask
                - 'uniformity_map': Uniformity score per pixel [0, 1]
                - 'uniform_regions': List of uniform region masks
        """
        H, W = image.shape[:2]
        
        # Heuristic segmentation using color clustering
        segmentation_map, class_masks = self._heuristic_segmentation(image)
        
        # Compute uniformity map
        uniformity_map = self._compute_uniformity(image)
        
        # Find uniform regions
        uniform_regions = self._find_uniform_regions(uniformity_map, threshold=0.7)
        
        return {
            'segmentation_map': segmentation_map,
            'class_masks': class_masks,
            'uniformity_map': uniformity_map,
            'uniform_regions': uniform_regions,
        }
    
    def _heuristic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Segment image using color and position heuristics.
        
        Heuristic rules:
        - Top region + blue/light -> sky
        - Middle region + vertical + uniform -> wall
        - Bottom region + horizontal -> floor/ground
        
        Args:
            image: Input image (H, W, 3) BGR
        
        Returns:
            (segmentation_map, class_masks)
        """
        H, W = image.shape[:2]
        
        # Convert to LAB for better color clustering
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Downsample for clustering
        scale = 0.25
        lab_small = cv2.resize(lab, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        H_s, W_s = lab_small.shape[:2]
        
        # Add position features
        y_coords = np.tile(np.linspace(0, 100, H_s)[:, None], (1, W_s))
        x_coords = np.tile(np.linspace(0, 100, W_s)[None, :], (H_s, 1))
        
        # Combine color and position
        features = np.stack([
            lab_small[:, :, 0],
            lab_small[:, :, 1],
            lab_small[:, :, 2],
            y_coords,
            x_coords
        ], axis=2).reshape(-1, 5)
        
        # K-means clustering
        n_clusters = 5
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        seg_small = labels.reshape(H_s, W_s).astype(np.uint8)
        
        # Resize to original size
        segmentation_map = cv2.resize(seg_small, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # Classify clusters into semantic classes (heuristic)
        class_masks = {}
        
        for label in range(n_clusters):
            mask = (segmentation_map == label).astype(np.uint8)
            region_color = image[mask > 0].mean(axis=0) if mask.sum() > 0 else np.array([0, 0, 0])
            
            # Get region position
            ys, xs = np.where(mask > 0)
            if len(ys) > 0:
                avg_y = ys.mean() / H
                
                # Classify based on position and color
                b, g, r = region_color
                
                # Sky: top region + blue/light
                if avg_y < 0.3 and (b > 150 or (b > g and b > r)):
                    class_name = 'sky'
                # Floor/ground: bottom region
                elif avg_y > 0.7:
                    class_name = 'floor'
                # Wall: middle + uniform
                elif 0.2 < avg_y < 0.8:
                    class_name = 'wall'
                else:
                    class_name = f'region_{label}'
                
                if class_name in class_masks:
                    class_masks[class_name] = cv2.bitwise_or(class_masks[class_name], mask * 255)
                else:
                    class_masks[class_name] = mask * 255
        
        return segmentation_map, class_masks
    
    def _compute_uniformity(self, image: np.ndarray, window_size: int = 31) -> np.ndarray:
        """
        Compute local uniformity (inverse of variance).
        
        Args:
            image: Input image (H, W, 3) BGR
            window_size: Window size for local statistics
        
        Returns:
            Uniformity map (H, W) float [0, 1], where 1=uniform
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compute local variance
        mean = cv2.boxFilter(gray, -1, (window_size, window_size))
        mean_sq = cv2.boxFilter(gray ** 2, -1, (window_size, window_size))
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)  # Numerical stability
        
        # Uniformity = inverse of variance (normalized)
        std = np.sqrt(variance)
        max_std = std.max() + 1e-8
        uniformity = 1.0 - (std / max_std)
        
        return uniformity.astype(np.float32)
    
    def _find_uniform_regions(
        self,
        uniformity_map: np.ndarray,
        threshold: float = 0.7,
        min_area: int = 1000
    ) -> List[np.ndarray]:
        """
        Find uniform regions above threshold.
        
        Args:
            uniformity_map: Uniformity map (H, W) float [0, 1]
            threshold: Uniformity threshold
            min_area: Minimum region area in pixels
        
        Returns:
            List of binary masks for uniform regions
        """
        # Threshold uniformity
        uniform_mask = (uniformity_map >= threshold).astype(np.uint8) * 255
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        uniform_mask = cv2.morphologyEx(uniform_mask, cv2.MORPH_CLOSE, kernel)
        uniform_mask = cv2.morphologyEx(uniform_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(uniform_mask, connectivity=8)
        
        uniform_regions = []
        for label in range(1, num_labels):  # Skip background (0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                region_mask = (labels == label).astype(np.uint8) * 255
                uniform_regions.append(region_mask)
        
        return uniform_regions


def test_segmentation():
    """Test semantic segmentation."""
    # Create test image with different regions
    img = np.ones((600, 800, 3), dtype=np.uint8)
    
    # Sky (top, blue)
    img[0:150, :] = [200, 180, 150]
    
    # Wall (middle, uniform gray)
    img[150:400, :] = [180, 180, 180]
    
    # Floor (bottom, darker)
    img[400:600, :] = [140, 140, 150]
    
    # Add some texture to wall
    noise = np.random.randint(-10, 10, (250, 800, 3), dtype=np.int16)
    img[150:400, :] = np.clip(img[150:400, :].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add objects (non-uniform)
    cv2.rectangle(img, (100, 200), (250, 350), (0, 0, 255), -1)
    cv2.circle(img, (650, 280), 70, (255, 200, 0), -1)
    
    segmenter = SemanticSegmenter()
    result = segmenter.segment(img)
    
    print(f"Segmentation map shape: {result['segmentation_map'].shape}")
    print(f"Unique segments: {len(np.unique(result['segmentation_map']))}")
    print(f"Semantic classes found: {list(result['class_masks'].keys())}")
    print(f"Uniformity range: [{result['uniformity_map'].min():.3f}, {result['uniformity_map'].max():.3f}]")
    print(f"Uniform regions found: {len(result['uniform_regions'])}")
    
    for i, region in enumerate(result['uniform_regions']):
        area = (region > 0).sum()
        print(f"  Region {i+1}: {area} pixels")
    
    return result


if __name__ == "__main__":
    test_segmentation()
