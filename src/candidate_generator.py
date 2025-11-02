"""Candidate generation for ad placement regions."""

import cv2
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .config import (
    MIN_AD_AREA_PCT,
    MAX_AD_AREA_PCT,
    ASPECT_RATIOS,
    MAX_RAW_CANDIDATES,
    MAX_FILTERED_CANDIDATES,
)

try:
    from .context_detector import ContextDetector, AdSurface
    CONTEXT_AVAILABLE = True
except ImportError:
    CONTEXT_AVAILABLE = False


@dataclass
class Candidate:
    """Ad placement candidate region."""
    x: int
    y: int
    w: int
    h: int
    score: float = 0.0
    reasons: List[str] = None
    
    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)
    
    @property
    def area(self) -> int:
        return self.w * self.h
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)


class CandidateGenerator:
    """Generate candidate regions for ad placement."""
    
    def __init__(self, use_context: bool = True, verbose: bool = False):
        """
        Initialize generator.
        
        Args:
            use_context: Whether to use context-aware surface detection
            verbose: Whether to print generation info
        """
        self.use_context = use_context and CONTEXT_AVAILABLE
        self.verbose = verbose
        
        if self.use_context:
            self.context_detector = ContextDetector(verbose=False)
            if self.verbose:
                print("✓ Context-aware generation enabled (screens, frames, walls)")
        else:
            self.context_detector = None
            if self.verbose:
                print("  Context detection disabled (random sampling only)")
    
    def generate(
        self,
        image_shape: Tuple[int, int],
        protected_mask: np.ndarray,
        saliency_map: np.ndarray,
        uniformity_map: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> List[Candidate]:
        """
        Generate candidate ad placement regions.
        
        Args:
            image_shape: (H, W) of image
            protected_mask: Binary mask of protected regions (H, W) uint8
            saliency_map: Saliency map (H, W) float [0, 1]
            uniformity_map: Optional uniformity score map (H, W)
            image: Optional original image for context detection
        
        Returns:
            List of Candidate objects (unranked, up to MAX_FILTERED_CANDIDATES)
        """
        H, W = image_shape
        total_area = H * W
        min_area = int(total_area * MIN_AD_AREA_PCT)
        max_area = int(total_area * MAX_AD_AREA_PCT)
        
        # Compute distance transform from protected regions
        dist_transform = cv2.distanceTransform(
            (~(protected_mask > 0)).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Inverse saliency (low saliency = better for ads)
        low_saliency = 1.0 - saliency_map
        
        candidates = []
        
        # Strategy 0: Context-aware surface detection (if available)
        if self.use_context and image is not None:
            surfaces = self.context_detector.detect_surfaces(image)
            
            if self.verbose:
                print(f"  Context detector found {len(surfaces)} ad-friendly surfaces")
            
            for surface in surfaces[:15]:  # Use top 15 detected surfaces
                x, y, w, h = surface.x, surface.y, surface.w, surface.h
                
                # Check if within bounds
                if x < 0 or y < 0 or x + w > W or y + h > H:
                    continue
                
                # Check if overlaps protected content
                region_protected = protected_mask[y:y+h, x:x+w]
                if region_protected.any():
                    continue
                
                # High base score for context-aware surfaces
                base_score = 0.7 + (surface.confidence * 0.3)
                
                # Bonus for screens (most natural ad placement)
                if surface.type == 'screen':
                    base_score = min(base_score + 0.15, 1.0)
                
                # Bonus for frames/posters
                elif surface.type in ['frame', 'wall_poster']:
                    base_score = min(base_score + 0.10, 1.0)
                
                candidates.append(Candidate(
                    x=x, y=y, w=w, h=h, score=base_score,
                    reasons=[f'context_{surface.type}', 'natural_placement']
                ))
            
            if self.verbose and len(surfaces) > 0:
                print(f"  Added {len([c for c in candidates if 'context_' in str(c.reasons)])} context-aware candidates")
        
        # Strategy 1: Random sampling with constraints
        for _ in range(MAX_RAW_CANDIDATES):
            # Pick random aspect ratio
            ar_w, ar_h = random.choice(ASPECT_RATIOS)
            
            # Pick random area
            area = random.uniform(min_area, max_area)
            
            # Compute dimensions
            w = int((area * ar_w / ar_h) ** 0.5)
            h = int(area / w) if w > 0 else 0
            
            if w <= 0 or h <= 0 or w > W or h > H:
                continue
            
            # Random position
            x = random.randint(0, max(1, W - w))
            y = random.randint(0, max(1, H - h))
            
            # Check constraints
            region_protected = protected_mask[y:y+h, x:x+w]
            if region_protected.any():
                continue  # Overlaps protected content
            
            # Compute score (basic heuristic)
            region_saliency = saliency_map[y:y+h, x:x+w]
            region_dist = dist_transform[y:y+h, x:x+w]
            
            score = (
                0.5 * (1.0 - region_saliency.mean()) +  # Low saliency better
                0.3 * (region_dist.mean() / max(H, W)) +  # Far from protected
                0.2 * (1.0 - abs(w / h - ar_w / ar_h))  # Aspect ratio match
            )
            
            candidates.append(Candidate(x=x, y=y, w=w, h=h, score=score))
        
        # Strategy 2: Grid-based sampling on low-saliency regions
        # (Skip for now to save time; use random sampling)
        
        # Filter and sort
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        # Apply NMS to reduce overlap
        candidates = self._non_max_suppression(candidates, iou_threshold=0.3)
        
        return candidates[:MAX_FILTERED_CANDIDATES]
    
    def _non_max_suppression(self, candidates: List[Candidate], iou_threshold: float = 0.3) -> List[Candidate]:
        """
        Apply non-maximum suppression to reduce overlapping candidates.
        
        Args:
            candidates: List of candidates (sorted by score desc)
            iou_threshold: IoU threshold for suppression
        
        Returns:
            Filtered list of candidates
        """
        if not candidates:
            return []
        
        keep = []
        candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
        
        while candidates:
            best = candidates.pop(0)
            keep.append(best)
            
            # Remove candidates with high IoU
            candidates = [
                c for c in candidates
                if self._compute_iou(best.bbox, c.bbox) < iou_threshold
            ]
        
        return keep
    
    def _compute_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Compute IoU between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-8)


def test_generator():
    """Test candidate generation."""
    H, W = 600, 800
    
    # Create mock data
    protected_mask = np.zeros((H, W), dtype=np.uint8)
    protected_mask[100:200, 100:300] = 255  # Protected region
    
    saliency_map = np.random.rand(H, W).astype(np.float32) * 0.5
    saliency_map[100:200, 100:300] = 0.9  # High saliency in protected
    
    generator = CandidateGenerator()
    candidates = generator.generate((H, W), protected_mask, saliency_map)
    
    print(f"Generated {len(candidates)} candidates")
    if candidates:
        print(f"Top candidate: x={candidates[0].x}, y={candidates[0].y}, "
              f"w={candidates[0].w}, h={candidates[0].h}, score={candidates[0].score:.3f}")
    
    return candidates


if __name__ == "__main__":
    test_generator()
