"""
Integrated TV screen detector and ad placer.
Uses context detection + deep learning person segmentation.
"""

import cv2
import numpy as np
from src.context_detector import ContextDetector
from typing import Optional, Tuple

try:
    from src.deep_segmentation import DeepPersonSegmenter
    DEEP_SEGMENTATION_AVAILABLE = True
except ImportError:
    DEEP_SEGMENTATION_AVAILABLE = False

try:
    from src.depth_estimation import DepthEstimator
    DEPTH_ESTIMATION_AVAILABLE = True
except ImportError:
    DEPTH_ESTIMATION_AVAILABLE = False

try:
    from src.edge_detection import PreciseEdgeDetector
    EDGE_DETECTION_AVAILABLE = True
except ImportError:
    EDGE_DETECTION_AVAILABLE = False

class PreciseScreenPlacer:
    """Precisely detect and place ads on TV screens."""
    
    def __init__(self, use_deep_learning: bool = True, use_depth: bool = True, 
                 use_edge_refinement: bool = True, device: str = 'cpu', verbose: bool = False):
        """
        Initialize placer.
        
        Args:
            use_deep_learning: Use DeepLabV3+ for person segmentation (more accurate)
            use_depth: Use MiDaS for depth estimation
            use_edge_refinement: Use precise edge detection for screen boundaries
            device: 'cpu' or 'cuda' for deep learning models
            verbose: Print debug info
        """
        self.verbose = verbose
        self.use_deep_learning = use_deep_learning and DEEP_SEGMENTATION_AVAILABLE
        self.use_depth = use_depth and DEPTH_ESTIMATION_AVAILABLE
        self.use_edge_refinement = use_edge_refinement and EDGE_DETECTION_AVAILABLE
        self.device = device
        self.context_detector = ContextDetector(verbose=False)
        
        # Initialize models
        self.person_segmenter = None
        self.depth_estimator = None
        self.edge_detector = None
        
        if self.verbose:
            print("Loading models...")
        
        if self.use_deep_learning:
            if self.verbose:
                print("  - DeepLabV3+ for person segmentation...")
            self.person_segmenter = DeepPersonSegmenter(device=device, verbose=verbose)
        
        if self.use_depth:
            if self.verbose:
                print("  - MiDaS for depth estimation...")
            self.depth_estimator = DepthEstimator(model_type="DPT_Large", device=device, verbose=verbose)
        
        if self.use_edge_refinement:
            if self.verbose:
                print("  - Precise edge detector...")
            self.edge_detector = PreciseEdgeDetector(verbose=verbose)
        
        if self.verbose:
            features = []
            if self.use_deep_learning:
                features.append("person segmentation")
            if self.use_depth:
                features.append("depth estimation")
            if self.use_edge_refinement:
                features.append("edge refinement")
            
            if features:
                print(f"✓ Precise screen placer initialized with: {', '.join(features)}")
            else:
                print("✓ Precise screen placer initialized (basic mode)")
    
    def place_ad_on_screen(self, image: np.ndarray, ad: np.ndarray, 
                          protected_mask: np.ndarray) -> Tuple[np.ndarray, bool, dict]:
        """
        Detect TV screen and place ad with proper depth handling.
        
        Returns:
            (result_image, success, info_dict)
        """
        H, W = image.shape[:2]
        
        # Step 1: Detect screen using context detector
        surfaces = self.context_detector.detect_surfaces(image)
        screen_surfaces = [s for s in surfaces if s.type == 'screen']
        
        if not screen_surfaces:
            if self.verbose:
                print("  No screens detected")
            return image, False, {'error': 'No screen found'}
        
        screen = screen_surfaces[0]  # Best screen
        x, y, w, h = screen.x, screen.y, screen.w, screen.h
        
        if self.verbose:
            print(f"  Screen found: ({x}, {y}) {w}x{h}")
        
        # Step 2: Refine screen boundaries using precise edge detection
        if self.use_edge_refinement and self.edge_detector:
            refined_bbox = self.edge_detector.detect_screen_edges(image, (x, y, w, h), expand_margin=30)
            x, y, w, h = refined_bbox
            if self.verbose:
                print(f"  Screen refined: ({x}, {y}) {w}x{h}")
        
        # Step 3: Estimate depth map
        depth_map = None
        if self.use_depth and self.depth_estimator:
            depth_map = self.depth_estimator.estimate_depth(image)
        
        # Step 4: Create depth-aware mask combining person segmentation + depth
        depth_mask = self._create_depth_mask(image, protected_mask, x, y, w, h, depth_map)
        
        # Step 4: Apply perspective transform and composite
        result = self._composite_with_perspective(
            image, ad, x, y, w, h, depth_mask
        )
        
        info = {
            'screen_bounds': (x, y, w, h),
            'confidence': screen.confidence
        }
        
        return result, True, info
    
    def _refine_screen_boundaries(self, image: np.ndarray, 
                                  x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """Refine screen boundaries using edge detection."""
        # Extract screen region with margin
        margin = 30
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)
        
        roi = image[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 30, 100)
        
        # Find strongest horizontal and vertical lines (screen borders)
        # Hough lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=min(w, h)//3, maxLineGap=20)
        
        if lines is not None and len(lines) > 0:
            # Find screen borders
            horizontal_lines = []
            vertical_lines = []
            
            for line in lines:
                x1_l, y1_l, x2_l, y2_l = line[0]
                angle = np.abs(np.arctan2(y2_l - y1_l, x2_l - x1_l))
                
                if angle < np.pi/4 or angle > 3*np.pi/4:  # Horizontal
                    horizontal_lines.append((y1_l + y2_l) // 2)
                else:  # Vertical
                    vertical_lines.append((x1_l + x2_l) // 2)
            
            if horizontal_lines and vertical_lines:
                # Get top/bottom/left/right borders
                top = min(horizontal_lines)
                bottom = max(horizontal_lines)
                left = min(vertical_lines)
                right = max(vertical_lines)
                
                # Convert back to original coordinates
                x_new = x1 + left
                y_new = y1 + top
                w_new = right - left
                h_new = bottom - top
                
                # Sanity check: not too different from original
                if (0.7 * w < w_new < 1.3 * w and 
                    0.7 * h < h_new < 1.3 * h):
                    if self.verbose:
                        print(f"  Refined boundaries: ({x_new}, {y_new}) {w_new}x{h_new}")
                    return x_new, y_new, w_new, h_new
        
        # If refinement fails, return original
        return x, y, w, h
    
    def _create_depth_mask(self, image: np.ndarray, protected_mask: np.ndarray,
                          x: int, y: int, w: int, h: int, depth_map: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create mask showing where ad can be placed (avoiding people).
        
        Combines person segmentation + depth estimation for best results.
        """
        H, W = image.shape[:2]
        
        # Start with full screen region
        screen_mask = np.zeros((H, W), dtype=np.uint8)
        screen_mask[y:y+h, x:x+w] = 255
        
        # Combine multiple masks
        final_mask = screen_mask.copy()
        person_mask = None
        
        # METHOD 1: Person segmentation (primary method)
        if self.use_deep_learning and self.person_segmenter is not None:
            if self.verbose:
                print(f"  Using DeepLabV3+ for person segmentation...")
            
            # Get person mask for entire image
            person_mask = self.person_segmenter.segment_people(image)
            person_mask = self.person_segmenter.refine_mask(person_mask, iterations=2)
            
            # Invert: we want background (not person)
            background_mask = cv2.bitwise_not(person_mask)
            
            # Apply to screen region
            final_mask = cv2.bitwise_and(final_mask, background_mask)
            
            if self.verbose:
                person_in_screen = np.sum(person_mask[y:y+h, x:x+w] > 127)
                total = w * h
                print(f"    Person in screen: {100*person_in_screen/total:.1f}%")
        
        # METHOD 2: Depth estimation (secondary - reinforces person segmentation)
        if self.use_depth and depth_map is not None:
            if self.verbose:
                print(f"  Using MiDaS depth for additional masking...")
            
            # Get depth mask for screen region
            depth_mask = self.depth_estimator.get_depth_mask(
                depth_map, x, y, w, h, threshold_percentile=40
            )
            
            # Combine with existing mask (intersection)
            final_mask = cv2.bitwise_and(final_mask, depth_mask)
            
            if self.verbose:
                depth_valid = np.sum(depth_mask[y:y+h, x:x+w] > 0)
                total = w * h
                print(f"    Depth-valid area: {100*depth_valid/total:.1f}%")
        
        # Final refinement in screen region
        screen_region_mask = final_mask[y:y+h, x:x+w]
        
        # Erode slightly to create safety margin from edges
        kernel = np.ones((3, 3), np.uint8)
        screen_region_mask = cv2.erode(screen_region_mask, kernel, iterations=1)
        
        final_mask[y:y+h, x:x+w] = screen_region_mask
        
        if self.verbose:
            screen_pixels = np.sum(final_mask[y:y+h, x:x+w] > 0)
            total = w * h
            print(f"    Final safe screen area: {100*screen_pixels/total:.1f}%")
        
        else:
            # METHOD 2: Color-based segmentation (fallback)
            if self.verbose:
                print(f"  Using color-based segmentation (fallback)...")
            
            # Extract screen region
            screen_region = image[y:y+h, x:x+w].copy()
            
            # The TV screen is blue and uniform
            # Convert to HSV
            hsv = cv2.cvtColor(screen_region, cv2.COLOR_BGR2HSV)
            
            # Detect blue screen pixels
            lower_blue = np.array([85, 40, 80])
            upper_blue = np.array([135, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Also detect bright pixels (screen glow)
            gray = cv2.cvtColor(screen_region, cv2.COLOR_BGR2GRAY)
            bright_mask = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)[1]
            
            # Combine: pixels that are blue OR bright are screen
            screen_pixels_mask = cv2.bitwise_or(blue_mask, bright_mask)
            
            # Clean up with morphology
            kernel = np.ones((5, 5), np.uint8)
            screen_pixels_mask = cv2.morphologyEx(screen_pixels_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            screen_pixels_mask = cv2.morphologyEx(screen_pixels_mask, cv2.MORPH_OPEN, kernel)
            
            # Apply to full image mask
            final_mask = screen_mask.copy()
            final_mask[y:y+h, x:x+w] = screen_pixels_mask
            
            # Erode slightly to avoid edge artifacts
            kernel_erode = np.ones((3, 3), np.uint8)
            final_mask = cv2.erode(final_mask, kernel_erode, iterations=2)
            
            if self.verbose:
                screen_pixels = np.sum(final_mask[y:y+h, x:x+w] > 0)
                total_screen_pixels = w * h
                coverage = 100 * screen_pixels / total_screen_pixels if total_screen_pixels > 0 else 0
                print(f"  Screen coverage: {coverage:.1f}%")
        
        # Smooth the mask for seamless blending
        final_mask = cv2.GaussianBlur(final_mask, (11, 11), 0)
        
        # Save debug mask
        cv2.imwrite('outputs/debug_screen_mask_deep.jpg', final_mask)
        
        return final_mask
    
    def _composite_with_perspective(self, image: np.ndarray, ad: np.ndarray,
                                   x: int, y: int, w: int, h: int,
                                   mask: np.ndarray) -> np.ndarray:
        """
        Composite ad onto screen with proper fitting (maintain aspect ratio).
        """
        H, W = image.shape[:2]
        
        # Calculate aspect ratios
        screen_aspect = w / h
        ad_aspect = ad.shape[1] / ad.shape[0]
        
        # Fit ad to screen size while maintaining aspect ratio
        if ad_aspect > screen_aspect:
            # Ad is wider - fit to width
            new_w = w
            new_h = int(w / ad_aspect)
        else:
            # Ad is taller - fit to height
            new_h = h
            new_w = int(h * ad_aspect)
        
        # Resize ad maintaining aspect ratio
        ad_resized = cv2.resize(ad, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Center the ad within the screen region
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2
        
        # Create result image
        result = image.copy()
        
        # Place ad in center of screen region
        ad_x = x + offset_x
        ad_y = y + offset_y
        
        # Extract mask for ad region
        mask_region = mask[ad_y:ad_y+new_h, ad_x:ad_x+new_w].astype(np.float32) / 255.0
        mask_region = np.expand_dims(mask_region, axis=2)
        
        # Blend ad onto screen region using mask
        screen_region = result[ad_y:ad_y+new_h, ad_x:ad_x+new_w].astype(np.float32)
        ad_float = ad_resized.astype(np.float32)
        
        blended = mask_region * ad_float + (1.0 - mask_region) * screen_region
        result[ad_y:ad_y+new_h, ad_x:ad_x+new_w] = blended.astype(np.uint8)
        
        if self.verbose:
            print(f"  Ad fitted: {new_w}x{new_h} centered at ({ad_x}, {ad_y})")
        
        return result


def demo_precise_placement():
    """Demo: Precise TV ad placement."""
    import os
    
    # Load image
    image_path = 'image.png'
    if not os.path.exists(image_path):
        image_path = 'examples/living_room.jpg'
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return
    
    # Load or create ad
    ad_path = 'examples/sample_ad.png'
    if os.path.exists(ad_path):
        ad = cv2.imread(ad_path)
    else:
        # Create simple ad
        ad = np.ones((400, 600, 3), dtype=np.uint8)
        ad[:] = (50, 50, 150)
        cv2.putText(ad, "NEW PRODUCT", (100, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        cv2.putText(ad, "AVAILABLE NOW", (120, 280), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    
    # Detect protected content
    from src.detectors import ProtectedContentDetector
    detector = ProtectedContentDetector()
    detection = detector.detect(image)
    protected_mask = detection['combined_mask']
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")
    print(f"Protected regions: {len(detection['bboxes'])}")
    
    # Place ad on screen (only use person segmentation, disable depth and edge refinement)
    placer = PreciseScreenPlacer(use_depth=False, use_edge_refinement=False, verbose=True)
    result, success, info = placer.place_ad_on_screen(image, ad, protected_mask)
    
    if success:
        print(f"\n✓ Ad placed successfully!")
        print(f"  Screen: {info['screen_bounds']}")
        print(f"  Confidence: {info['confidence']:.1%}")
        
        cv2.imwrite('outputs/precise_tv_placement.jpg', result)
        print(f"\n✓ Saved to: outputs/precise_tv_placement.jpg")
    else:
        print(f"\n✗ Placement failed: {info.get('error', 'Unknown error')}")


if __name__ == "__main__":
    demo_precise_placement()
