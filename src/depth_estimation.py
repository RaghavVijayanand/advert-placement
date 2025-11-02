"""
Monocular depth estimation using MiDaS model for depth-aware ad placement.
"""
import cv2
import numpy as np
import torch
import time
from typing import Tuple, Optional

try:
    import torch.hub
    MIDAS_AVAILABLE = True
except ImportError:
    MIDAS_AVAILABLE = False


class DepthEstimator:
    """Estimate depth using MiDaS model for depth-aware compositing."""
    
    def __init__(self, model_type: str = "DPT_Large", device: str = 'cpu', verbose: bool = False):
        """
        Initialize depth estimator.
        
        Args:
            model_type: Model variant - "DPT_Large" (best), "DPT_Hybrid", "MiDaS_small" (fastest)
            device: 'cpu' or 'cuda'
            verbose: Print progress
        """
        self.device = torch.device(device)
        self.verbose = verbose
        self.model = None
        self.transform = None
        
        if not MIDAS_AVAILABLE:
            raise ImportError("torch is required for depth estimation")
        
        if self.verbose:
            print(f"Loading MiDaS {model_type} model on {device}...")
        
        t0 = time.time()
        
        # Load MiDaS model from torch hub
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform
        
        if self.verbose:
            print(f"✓ MiDaS model loaded in {time.time() - t0:.2f}s")
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for image.
        
        Args:
            image: BGR image (H, W, 3)
        
        Returns:
            Depth map (H, W) - higher values = further away, lower = closer
        """
        if self.verbose:
            print(f"  Estimating depth for {image.shape[1]}x{image.shape[0]} image...")
        
        t0 = time.time()
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = self.transform(img_rgb).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.model(input_batch)
            
            # Resize to original resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-255 for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        
        if self.verbose:
            print(f"  ✓ Depth estimation complete in {time.time() - t0:.2f}s")
            print(f"    Depth range: {depth_min:.2f} to {depth_max:.2f}")
        
        return depth_normalized
    
    def get_depth_mask(self, depth_map: np.ndarray, x: int, y: int, w: int, h: int, 
                      threshold_percentile: float = 50) -> np.ndarray:
        """
        Create mask for pixels at similar depth to screen region.
        
        Args:
            depth_map: Normalized depth map (0-255)
            x, y, w, h: Screen bounding box
            threshold_percentile: Percentile for depth threshold (50 = median)
        
        Returns:
            Binary mask (H, W) - 255 for screen-depth pixels, 0 for closer objects
        """
        H, W = depth_map.shape
        
        # Get depth values in screen region
        screen_depth = depth_map[y:y+h, x:x+w]
        
        # Calculate threshold (median depth of screen)
        depth_threshold = np.percentile(screen_depth, threshold_percentile)
        
        if self.verbose:
            print(f"    Screen depth threshold: {depth_threshold:.1f}")
        
        # Pixels at screen depth or further are valid
        # (in MiDaS, higher values = further away)
        depth_mask = (depth_map >= depth_threshold).astype(np.uint8) * 255
        
        return depth_mask


def test_depth_estimation():
    """Test depth estimation on living room image."""
    print("=" * 60)
    print("Testing MiDaS Depth Estimation")
    print("=" * 60)
    
    # Load test image
    image_path = 'examples/living_room.jpg'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load {image_path}")
        return
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")
    print(f"Using device: cpu")
    
    # Initialize depth estimator
    estimator = DepthEstimator(model_type="DPT_Large", device='cpu', verbose=True)
    
    # Estimate depth
    depth_map = estimator.estimate_depth(image)
    
    # Save depth visualization
    depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    cv2.imwrite('outputs/depth_map.jpg', depth_colored)
    print(f"✓ Depth map saved to outputs/depth_map.jpg")
    
    # Test with known screen region (from context detector)
    x, y, w, h = 49, 180, 420, 267
    
    # Get depth mask
    depth_mask = estimator.get_depth_mask(depth_map, x, y, w, h, threshold_percentile=40)
    
    # Visualize depth mask
    cv2.imwrite('outputs/depth_mask.jpg', depth_mask)
    print(f"✓ Depth mask saved to outputs/depth_mask.jpg")
    
    # Show depth mask on screen region
    overlay = image.copy()
    mask_colored = cv2.cvtColor(depth_mask, cv2.COLOR_GRAY2BGR)
    mask_colored[:, :, 1] = 0  # Remove green, keep blue/red
    overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
    
    # Draw screen bbox
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite('outputs/depth_overlay.jpg', overlay)
    print(f"✓ Depth overlay saved to outputs/depth_overlay.jpg")
    
    # Stats
    screen_mask_pixels = np.sum(depth_mask[y:y+h, x:x+w] > 0)
    total_screen = w * h
    coverage = 100 * screen_mask_pixels / total_screen
    print(f"\nDepth-based screen coverage: {coverage:.1f}%")


if __name__ == "__main__":
    test_depth_estimation()
