"""Saliency estimation using pretrained models."""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

from .config import SALIENCY_THRESHOLD


class SaliencyEstimator:
    """Estimates visual saliency using lightweight pretrained model."""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize saliency model.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize saliency model (placeholder for U^2-Net or similar)."""
        # For now, use a simple heuristic-based approach
        # In production: load U^2-Net, BASNet, or similar from torch.hub
        print("✓ Saliency estimator initialized (heuristic mode)")
    
    def estimate(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate saliency map for input image.
        
        Args:
            image: Input image (H, W, 3) BGR
        
        Returns:
            Saliency map (H, W) float32, normalized to [0, 1]
            Higher values = more salient (attention-grabbing)
        """
        # Heuristic-based saliency (fast, no deep learning)
        saliency_map = self._heuristic_saliency(image)
        
        # Normalize to [0, 1]
        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        
        return saliency_map.astype(np.float32)
    
    def _heuristic_saliency(self, image: np.ndarray) -> np.ndarray:
        """
        Compute saliency using spectral residual + edge density heuristic.
        
        Args:
            image: Input image (H, W, 3) BGR
        
        Returns:
            Raw saliency map (H, W) float
        """
        H, W = image.shape[:2]
        
        # 1. Spectral Residual (frequency domain)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_float = gray.astype(np.float32) / 255.0
        
        # FFT
        fft = np.fft.fft2(gray_float)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Log magnitude
        log_mag = np.log(magnitude + 1e-8)
        
        # Spectral residual
        avg_log_mag = cv2.boxFilter(log_mag, -1, (3, 3))
        residual = log_mag - avg_log_mag
        
        # Reconstruct
        saliency_freq = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
        saliency_freq = cv2.GaussianBlur(saliency_freq, (11, 11), 0)
        
        # 2. Edge density (spatial)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.GaussianBlur(edges.astype(np.float32), (21, 21), 0)
        
        # 3. Color contrast (LAB color space)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]
        b_channel = lab[:, :, 2]
        
        # Compute local std dev for each channel
        l_std = self._local_std(l_channel, ksize=31)
        a_std = self._local_std(a_channel, ksize=31)
        b_std = self._local_std(b_channel, ksize=31)
        color_contrast = l_std + a_std + b_std
        
        # 4. Center bias (images tend to have salient content in center)
        y_coords, x_coords = np.ogrid[:H, :W]
        center_y, center_x = H / 2, W / 2
        sigma = min(H, W) / 3
        center_bias = np.exp(-((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2) / (2 * sigma ** 2))
        
        # Combine cues
        saliency = (
            0.4 * saliency_freq +
            0.3 * edge_density / (edge_density.max() + 1e-8) +
            0.2 * color_contrast / (color_contrast.max() + 1e-8) +
            0.1 * center_bias
        )
        
        return saliency
    
    def _local_std(self, channel: np.ndarray, ksize: int = 31) -> np.ndarray:
        """Compute local standard deviation."""
        mean = cv2.GaussianBlur(channel, (ksize, ksize), 0)
        mean_sq = cv2.GaussianBlur(channel ** 2, (ksize, ksize), 0)
        variance = mean_sq - mean ** 2
        variance = np.maximum(variance, 0)  # Numerical stability
        return np.sqrt(variance)
    
    def get_high_saliency_mask(self, saliency_map: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Get binary mask of high-saliency regions.
        
        Args:
            saliency_map: Saliency map (H, W) float [0, 1]
            threshold: Threshold value (default from config)
        
        Returns:
            Binary mask (H, W) uint8, 255=high saliency
        """
        thresh = threshold or SALIENCY_THRESHOLD
        high_sal_mask = (saliency_map >= thresh).astype(np.uint8) * 255
        return high_sal_mask


def test_saliency():
    """Test saliency estimation on a sample image."""
    # Create test image with varying content
    img = np.ones((600, 800, 3), dtype=np.uint8) * 200
    
    # Add high-contrast region (should be salient)
    cv2.rectangle(img, (300, 200), (500, 400), (0, 0, 255), -1)
    cv2.circle(img, (650, 150), 80, (255, 255, 0), -1)
    
    # Add text (should be salient)
    cv2.putText(img, "ATTENTION", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    
    estimator = SaliencyEstimator()
    saliency_map = estimator.estimate(img)
    high_sal_mask = estimator.get_high_saliency_mask(saliency_map)
    
    print(f"Saliency map shape: {saliency_map.shape}")
    print(f"Saliency range: [{saliency_map.min():.3f}, {saliency_map.max():.3f}]")
    print(f"High-saliency coverage: {(high_sal_mask > 0).sum() / high_sal_mask.size * 100:.1f}%")
    
    return saliency_map, high_sal_mask


if __name__ == "__main__":
    test_saliency()
