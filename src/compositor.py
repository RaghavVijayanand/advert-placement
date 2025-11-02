"""Ad compositing with perspective correction and validation."""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path

from .candidate_generator import Candidate
from .config import MIN_CONTRAST_RATIO, DEFAULT_SHADOW_OPACITY, DEFAULT_BORDER_WIDTH


class AdCompositor:
    """Composite ad assets onto poster images with validation."""
    
    def __init__(self):
        """Initialize compositor."""
        pass
    
    def composite(
        self,
        poster: np.ndarray,
        ad_asset: np.ndarray,
        candidate: Candidate,
        protected_mask: np.ndarray,
        add_border: bool = True,
        add_shadow: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Composite ad asset onto poster at candidate region.
        
        Args:
            poster: Poster image (H, W, 3) BGR
            ad_asset: Ad image (H_ad, W_ad, 3 or 4) BGR or BGRA
            candidate: Placement candidate
            protected_mask: Protected regions mask
            add_border: Add border if low contrast
            add_shadow: Add drop shadow
        
        Returns:
            Dict with:
                - 'composite': Final composited image
                - 'valid': Boolean, whether placement is valid
                - 'warnings': List of warning messages
        """
        x, y, w, h = candidate.bbox
        H_poster, W_poster = poster.shape[:2]
        
        warnings = []
        
        # Check if this is a screen/TV placement (no margins, respects depth)
        is_screen = any('context_screen' in str(reason) for reason in (candidate.reasons or []))
        
        # 1. Resize ad to fit region
        if is_screen:
            # For screens: fill completely, no margin
            margin = 0
            target_w = w
            target_h = h
            warnings.append('Screen detected: filling entire display')
        else:
            # For other placements: use margin
            margin = 10
            target_w = w - 2 * margin
            target_h = h - 2 * margin
        
        if target_w <= 0 or target_h <= 0:
            return {'composite': poster, 'valid': False, 'warnings': ['Region too small']}
        
        ad_resized = cv2.resize(ad_asset, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        # 2. Extract alpha channel if present
        if ad_resized.shape[2] == 4:
            ad_rgb = ad_resized[:, :, :3]
            ad_alpha = ad_resized[:, :, 3:4] / 255.0
        else:
            ad_rgb = ad_resized
            ad_alpha = np.ones((target_h, target_w, 1), dtype=np.float32)
        
        # 3. Calculate placement coordinates
        x_start = x + margin
        y_start = y + margin
        x_end = x_start + target_w
        y_end = y_start + target_h
        
        if x_end > W_poster or y_end > H_poster:
            return {'composite': poster, 'valid': False, 'warnings': ['Ad extends beyond image']}
        
        # For screens: Create a mask of where to place (avoid overlaying people)
        if is_screen:
            # Get the region where the ad will go
            placement_region = poster[y_start:y_end, x_start:x_end]
            
            # Create a mask: only place ad where there's no protected content
            placement_protected = protected_mask[y_start:y_end, x_start:x_end]
            
            # Invert to get valid placement mask (where we CAN place)
            placement_valid_mask = (placement_protected == 0).astype(np.float32)
            
            # Dilate protected areas slightly to avoid edge artifacts
            kernel = np.ones((5, 5), np.uint8)
            placement_protected_dilated = cv2.dilate(placement_protected, kernel, iterations=1)
            placement_valid_mask = (placement_protected_dilated == 0).astype(np.float32)
            
            # Smooth the mask to avoid hard edges
            placement_valid_mask = cv2.GaussianBlur(placement_valid_mask, (11, 11), 0)
            placement_valid_mask = np.expand_dims(placement_valid_mask, axis=2)  # Add channel dimension
        else:
            # For non-screens: check overlap normally
            placement_protected = protected_mask[y_start:y_end, x_start:x_end]
            if placement_protected.any():
                return {'composite': poster, 'valid': False, 'warnings': ['Overlaps protected content']}
            placement_valid_mask = None
        
        # 4. Check contrast and add border if needed
        background_region = poster[y_start:y_end, x_start:x_end]
        contrast_ok, needs_border = self._check_contrast(ad_rgb, background_region)
        
        if needs_border and add_border:
            ad_rgb_bordered = self._add_border(ad_rgb, width=DEFAULT_BORDER_WIDTH)
            # Resize back to original dimensions
            ad_rgb = cv2.resize(ad_rgb_bordered, (target_w, target_h), interpolation=cv2.INTER_AREA)
            warnings.append('Added border for legibility')
        
        # 5. Add drop shadow if requested
        composite = poster.copy()
        if add_shadow:
            composite = self._add_shadow(composite, x_start, y_start, target_w, target_h)
        
        # 6. Alpha blend ad onto composite
        bg_region = composite[y_start:y_end, x_start:x_end].astype(np.float32)
        ad_float = ad_rgb.astype(np.float32)
        
        if is_screen and placement_valid_mask is not None:
            # For screens: blend only where there's no person
            # This makes the ad appear BEHIND people (on the screen)
            final_alpha = ad_alpha * placement_valid_mask
            blended = final_alpha * ad_float + (1.0 - final_alpha) * bg_region
        else:
            # Normal blending
            blended = ad_alpha * ad_float + (1.0 - ad_alpha) * bg_region
        
        composite[y_start:y_end, x_start:x_end] = blended.astype(np.uint8)
        
        return {
            'composite': composite,
            'valid': True,
            'warnings': warnings,
        }
    
    def _check_contrast(self, ad: np.ndarray, background: np.ndarray) -> Tuple[bool, bool]:
        """
        Check if ad has sufficient contrast with background.
        
        Returns:
            (contrast_ok, needs_border)
        """
        # Simplified WCAG contrast check using luminance
        ad_lum = 0.299 * ad[:, :, 2] + 0.587 * ad[:, :, 1] + 0.114 * ad[:, :, 0]
        bg_lum = 0.299 * background[:, :, 2] + 0.587 * background[:, :, 1] + 0.114 * background[:, :, 0]
        
        ad_mean_lum = ad_lum.mean()
        bg_mean_lum = bg_lum.mean()
        
        # Contrast ratio (simplified)
        lighter = max(ad_mean_lum, bg_mean_lum) + 0.05
        darker = min(ad_mean_lum, bg_mean_lum) + 0.05
        contrast_ratio = lighter / darker
        
        contrast_ok = contrast_ratio >= MIN_CONTRAST_RATIO
        needs_border = not contrast_ok
        
        return contrast_ok, needs_border
    
    def _add_border(self, ad: np.ndarray, width: int = 2, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Add border around ad for legibility."""
        bordered = cv2.copyMakeBorder(
            ad,
            width, width, width, width,
            cv2.BORDER_CONSTANT,
            value=color
        )
        return bordered
    
    def _add_shadow(self, image: np.ndarray, x: int, y: int, w: int, h: int, opacity: float = DEFAULT_SHADOW_OPACITY) -> np.ndarray:
        """Add drop shadow behind ad region."""
        shadow = image.copy()
        offset = 5
        
        # Create shadow mask
        shadow_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(
            shadow_mask,
            (x + offset, y + offset),
            (x + w + offset, y + h + offset),
            255,
            -1
        )
        
        # Blur shadow
        shadow_mask = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
        
        # Apply shadow
        shadow_alpha = (shadow_mask / 255.0 * opacity).astype(np.float32)
        for c in range(3):
            shadow[:, :, c] = (
                (1.0 - shadow_alpha) * image[:, :, c] +
                shadow_alpha * 0  # Black shadow
            ).astype(np.uint8)
        
        return shadow


def test_compositor():
    """Test compositor on sample data."""
    # Create test poster
    poster = np.ones((600, 800, 3), dtype=np.uint8) * 200
    cv2.rectangle(poster, (100, 100), (300, 200), (150, 150, 255), -1)
    
    # Create test ad
    ad = np.ones((100, 150, 3), dtype=np.uint8) * 50
    cv2.putText(ad, "AD", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Candidate region
    from .candidate_generator import Candidate
    candidate = Candidate(x=400, y=300, w=200, h=150)
    
    # Protected mask
    protected = np.zeros((600, 800), dtype=np.uint8)
    protected[100:200, 100:300] = 255
    
    compositor = AdCompositor()
    result = compositor.composite(poster, ad, candidate, protected)
    
    print(f"Composite valid: {result['valid']}")
    print(f"Warnings: {result['warnings']}")
    
    return result


if __name__ == "__main__":
    test_compositor()
