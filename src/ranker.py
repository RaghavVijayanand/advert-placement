"""Candidate ranking with heuristic scoring."""

import numpy as np
import cv2
from typing import List, Tuple
from .candidate_generator import Candidate


class CandidateRanker:
    """Rank candidates using heuristic scoring."""
    
    def __init__(self):
        """Initialize ranker."""
        pass
    
    def rank(
        self,
        candidates: List[Candidate],
        image_shape: Tuple[int, int],
        saliency_map: np.ndarray,
        protected_mask: np.ndarray,
    ) -> List[Candidate]:
        """
        Rank candidates and add reasons.
        
        Args:
            candidates: List of candidates to rank
            image_shape: (H, W)
            saliency_map: Saliency map (H, W) float [0, 1]
            protected_mask: Protected mask (H, W) uint8
        
        Returns:
            Sorted list of candidates (best first) with updated scores and reasons
        """
        H, W = image_shape
        
        # Compute distance transform once
        dist_transform = cv2.distanceTransform(
            (~(protected_mask > 0)).astype(np.uint8),
            cv2.DIST_L2,
            5
        )
        
        # Thirds lines for composition
        third_x = [W / 3, 2 * W / 3]
        third_y = [H / 3, 2 * H / 3]
        
        for cand in candidates:
            x, y, w, h = cand.bbox
            
            # Extract regions
            region_saliency = saliency_map[y:y+h, x:x+w]
            region_dist = dist_transform[y:y+h, x:x+w]
            
            # Feature computation
            sal_mean = region_saliency.mean()
            sal_max = region_saliency.max()
            dist_mean = region_dist.mean() / max(H, W)  # Normalized
            
            # Edge distance (distance to image edges)
            edge_dist = min(x, y, W - (x + w), H - (y + h)) / max(H, W)
            
            # Composition score (distance to rule of thirds)
            cx, cy = cand.center
            dist_to_thirds = min(
                abs(cx - third_x[0]), abs(cx - third_x[1]),
                abs(cy - third_y[0]), abs(cy - third_y[1])
            ) / max(H, W)
            comp_score = 1.0 - dist_to_thirds  # Closer to thirds = better
            
            # Aspect ratio preference (prefer common ad formats)
            aspect = w / (h + 1e-8)
            aspect_prefs = [1.0, 4/3, 3/2, 16/9, 9/16]
            aspect_score = max([1.0 - abs(aspect - pref) / pref for pref in aspect_prefs])
            
            # Combined score
            score = (
                0.35 * (1.0 - sal_mean) +      # Low saliency overlap
                0.25 * dist_mean +              # Far from protected
                0.15 * edge_dist +              # Not too close to edges
                0.15 * comp_score +             # Near thirds
                0.10 * aspect_score             # Standard aspect
            )
            
            # Bonus for context-aware candidates
            context_bonus = 0.0
            if cand.reasons:
                for reason in cand.reasons:
                    if reason == 'context_screen':
                        context_bonus = 0.25  # Big bonus for TV screens
                    elif reason == 'context_frame':
                        context_bonus = 0.15  # Medium bonus for picture frames
                    elif reason == 'context_wall_poster':
                        context_bonus = 0.12  # Bonus for wall poster areas
                    elif reason == 'context_billboard':
                        context_bonus = 0.20  # Bonus for billboards
            
            score = min(score + context_bonus, 1.0)
            cand.score = score
            
            # Generate reasons (preserve existing context reasons)
            reasons = list(cand.reasons) if cand.reasons else []
            if sal_mean < 0.3:
                reasons.append("low_saliency_overlap")
            if dist_mean > 0.1:
                reasons.append("safe_distance_from_protected")
            if edge_dist > 0.05:
                reasons.append("not_edge_clipped")
            if comp_score > 0.7:
                reasons.append("near_rule_of_thirds")
            
            cand.reasons = reasons
        
        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)
        
        return candidates


def test_ranker():
    """Test candidate ranking."""
    from .candidate_generator import Candidate
    
    # Create mock candidates
    candidates = [
        Candidate(x=100, y=100, w=150, h=100),
        Candidate(x=400, y=300, w=200, h=150),
        Candidate(x=50, y=50, w=100, h=100),
    ]
    
    H, W = 600, 800
    saliency_map = np.random.rand(H, W).astype(np.float32) * 0.5
    protected_mask = np.zeros((H, W), dtype=np.uint8)
    protected_mask[100:150, 100:200] = 255
    
    ranker = CandidateRanker()
    ranked = ranker.rank(candidates, (H, W), saliency_map, protected_mask)
    
    print(f"Ranked {len(ranked)} candidates")
    for i, cand in enumerate(ranked[:3]):
        print(f"  #{i+1}: score={cand.score:.3f}, reasons={cand.reasons}")
    
    return ranked


if __name__ == "__main__":
    from typing import Tuple
    test_ranker()
