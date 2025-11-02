"""End-to-end pipeline for ad placement analysis and compositing."""

import cv2
import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .detectors import ProtectedContentDetector
from .saliency import SaliencyEstimator
from .candidate_generator import CandidateGenerator, Candidate
from .ranker import CandidateRanker
from .compositor import AdCompositor
from .config import TOP_K_CANDIDATES


class AdPlacementPipeline:
    """Complete pipeline for discreet ad placement."""
    
    def __init__(self, device: str = "cpu", verbose: bool = False):
        """
        Initialize pipeline with all components.
        
        Args:
            device: 'cpu' or 'cuda'
            verbose: Print progress messages
        """
        self.device = device
        self.verbose = verbose
        
        # Initialize components
        self._log("Initializing pipeline components...")
        self.detector = ProtectedContentDetector(device=device)
        self.saliency_estimator = SaliencyEstimator(device=device)
        self.candidate_generator = CandidateGenerator()
        self.ranker = CandidateRanker()
        self.compositor = AdCompositor()
        self._log("✓ Pipeline ready")
    
    def analyze(self, poster_path: str) -> Dict:
        """
        Analyze poster and return top candidate placements.
        
        Args:
            poster_path: Path to poster image
        
        Returns:
            Dict with keys:
                - 'candidates': List of top candidates
                - 'protected_mask': Protected regions mask
                - 'saliency_map': Saliency map
                - 'image': Original poster
                - 'timing': Processing times
        """
        timing = {}
        
        # Load image
        t0 = time.time()
        poster = cv2.imread(str(poster_path))
        if poster is None:
            raise ValueError(f"Failed to load image: {poster_path}")
        
        H, W = poster.shape[:2]
        self._log(f"Loaded poster: {W}x{H}")
        timing['load'] = time.time() - t0
        
        # 1. Detect protected content
        t0 = time.time()
        self._log("Detecting protected content...")
        detection_result = self.detector.detect(poster)
        protected_mask = detection_result['combined_mask']
        self._log(f"  Found {len(detection_result['bboxes'])} protected regions")
        timing['detection'] = time.time() - t0
        
        # 2. Estimate saliency
        t0 = time.time()
        self._log("Estimating saliency...")
        saliency_map = self.saliency_estimator.estimate(poster)
        timing['saliency'] = time.time() - t0
        
        # 3. Generate candidates
        t0 = time.time()
        self._log("Generating candidates...")
        candidates = self.candidate_generator.generate(
            (H, W),
            protected_mask,
            saliency_map,
            image=poster  # Pass image for context-aware detection
        )
        self._log(f"  Generated {len(candidates)} candidates")
        timing['generation'] = time.time() - t0
        
        # 4. Rank candidates
        t0 = time.time()
        self._log("Ranking candidates...")
        ranked_candidates = self.ranker.rank(
            candidates,
            (H, W),
            saliency_map,
            protected_mask
        )
        top_candidates = ranked_candidates[:TOP_K_CANDIDATES]
        self._log(f"  Top {len(top_candidates)} candidates selected")
        timing['ranking'] = time.time() - t0
        
        timing['total'] = sum(timing.values())
        self._log(f"✓ Analysis complete in {timing['total']:.2f}s")
        
        return {
            'candidates': top_candidates,
            'protected_mask': protected_mask,
            'saliency_map': saliency_map,
            'image': poster,
            'timing': timing,
        }
    
    def compose(
        self,
        poster: np.ndarray,
        ad_asset_path: str,
        candidate: Candidate,
        protected_mask: np.ndarray,
    ) -> Dict:
        """
        Composite ad onto poster at selected candidate.
        
        Args:
            poster: Poster image
            ad_asset_path: Path to ad image
            candidate: Selected candidate
            protected_mask: Protected regions mask
        
        Returns:
            Dict with keys:
                - 'composite': Final composited image
                - 'valid': Whether placement passed validation
                - 'warnings': List of warnings
        """
        self._log(f"Compositing ad at candidate (x={candidate.x}, y={candidate.y}, w={candidate.w}, h={candidate.h})...")
        
        # Load ad asset
        ad_asset = cv2.imread(str(ad_asset_path), cv2.IMREAD_UNCHANGED)
        if ad_asset is None:
            raise ValueError(f"Failed to load ad asset: {ad_asset_path}")
        
        # Composite
        result = self.compositor.composite(
            poster,
            ad_asset,
            candidate,
            protected_mask,
            add_border=True,
            add_shadow=True,
        )
        
        if result['valid']:
            self._log("✓ Composite successful")
        else:
            self._log(f"✗ Composite failed: {result['warnings']}")
        
        return result
    
    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)


def demo_pipeline():
    """Demo the complete pipeline."""
    # Create test poster
    test_poster_path = Path("outputs/test_poster.jpg")
    test_poster_path.parent.mkdir(exist_ok=True)
    
    poster = np.ones((600, 800, 3), dtype=np.uint8) * 220
    # Add content
    cv2.rectangle(poster, (100, 100), (300, 250), (180, 180, 255), -1)
    cv2.putText(poster, "CONCERT", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 4)
    cv2.circle(poster, (650, 150), 60, (255, 200, 100), -1)
    cv2.imwrite(str(test_poster_path), poster)
    
    # Create test ad
    test_ad_path = Path("outputs/test_ad.png")
    ad = np.ones((100, 150, 3), dtype=np.uint8) * 40
    cv2.putText(ad, "BUY NOW", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imwrite(str(test_ad_path), ad)
    
    # Run pipeline
    pipeline = AdPlacementPipeline(verbose=True)
    
    print("\n=== ANALYSIS PHASE ===")
    analysis = pipeline.analyze(str(test_poster_path))
    
    print(f"\nTop {len(analysis['candidates'])} candidates:")
    for i, cand in enumerate(analysis['candidates']):
        print(f"  #{i+1}: (x={cand.x}, y={cand.y}, w={cand.w}, h={cand.h}) "
              f"score={cand.score:.3f} reasons={cand.reasons}")
    
    print("\n=== COMPOSITION PHASE ===")
    if analysis['candidates']:
        selected = analysis['candidates'][0]
        print(f"User selects candidate #1")
        
        result = pipeline.compose(
            analysis['image'],
            str(test_ad_path),
            selected,
            analysis['protected_mask']
        )
        
        if result['valid']:
            output_path = Path("outputs/final_composite.jpg")
            cv2.imwrite(str(output_path), result['composite'])
            print(f"\n✓ Saved composite to {output_path}")
        else:
            print(f"\n✗ Composite invalid: {result['warnings']}")
    
    print(f"\nTotal time: {analysis['timing']['total']:.2f}s")


if __name__ == "__main__":
    demo_pipeline()
