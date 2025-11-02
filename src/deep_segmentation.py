"""
Deep learning-based person segmentation for accurate foreground/background separation.

Uses DeepLabV3+ with ResNet101 backbone for semantic segmentation.
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from typing import Tuple, Optional
import time


class DeepPersonSegmenter:
    """Deep learning person segmentation using DeepLabV3+."""
    
    def __init__(self, device: str = 'cpu', verbose: bool = False):
        """
        Initialize segmenter with DeepLabV3+ model.
        
        Args:
            device: 'cpu' or 'cuda'
            verbose: Print timing and debug info
        """
        self.device = device
        self.verbose = verbose
        
        if self.verbose:
            print(f"Loading DeepLabV3+ ResNet101 model on {device}...")
            t0 = time.time()
        
        # Load pretrained DeepLabV3+ model
        weights = DeepLabV3_ResNet101_Weights.DEFAULT
        self.model = deeplabv3_resnet101(weights=weights)
        self.model.to(device)
        self.model.eval()
        
        # Get transforms
        self.transforms = weights.transforms()
        
        # COCO class IDs for people
        self.person_class = 15  # Person class in COCO
        
        if self.verbose:
            print(f"✓ Model loaded in {time.time() - t0:.2f}s")
    
    def segment_people(self, image: np.ndarray) -> np.ndarray:
        """
        Segment people from image.
        
        Args:
            image: Input image (H, W, 3) BGR
            
        Returns:
            Binary mask (H, W) uint8 where 255 = person, 0 = background
        """
        H, W = image.shape[:2]
        
        if self.verbose:
            print(f"  Segmenting people in {W}x{H} image...")
            t0 = time.time()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL for transforms
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms and add batch dimension
        input_tensor = self.transforms(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)['out'][0]
        
        # Get class predictions
        predictions = output.argmax(0).cpu().numpy()
        
        # Create person mask
        person_mask = (predictions == self.person_class).astype(np.uint8) * 255
        
        # Resize back to original size
        person_mask = cv2.resize(person_mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        if self.verbose:
            person_pixels = np.sum(person_mask > 0)
            total_pixels = H * W
            print(f"  ✓ Segmentation complete in {time.time() - t0:.2f}s")
            print(f"    Person pixels: {person_pixels} ({100*person_pixels/total_pixels:.1f}%)")
        
        return person_mask
    
    def refine_mask(self, mask: np.ndarray, iterations: int = 2) -> np.ndarray:
        """
        Refine mask with morphological operations.
        
        Args:
            mask: Binary mask
            iterations: Number of refinement iterations
            
        Returns:
            Refined mask
        """
        # Close small holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Smooth edges with Gaussian blur
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        
        return mask


def test_deep_segmentation():
    """Test deep learning segmentation."""
    import os
    
    # Load image
    image_path = 'examples/living_room.jpg'
    if not os.path.exists(image_path):
        image_path = 'image.png'
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return
    
    print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")
    
    # Detect GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Segment
    segmenter = DeepPersonSegmenter(device=device, verbose=True)
    person_mask = segmenter.segment_people(image)
    
    # Refine
    print("\nRefining mask...")
    person_mask_refined = segmenter.refine_mask(person_mask)
    
    # Visualize
    # 1. Save mask
    cv2.imwrite('outputs/person_mask_deep.jpg', person_mask_refined)
    print(f"\n✓ Person mask saved to outputs/person_mask_deep.jpg")
    
    # 2. Overlay on image
    overlay = image.copy()
    overlay[person_mask_refined > 127] = [0, 255, 0]  # Green for person
    result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    cv2.imwrite('outputs/person_overlay_deep.jpg', result)
    print(f"✓ Overlay saved to outputs/person_overlay_deep.jpg")
    
    # 3. Extract person
    person_only = image.copy()
    person_only[person_mask_refined < 127] = [255, 255, 255]  # White background
    cv2.imwrite('outputs/person_extracted_deep.jpg', person_only)
    print(f"✓ Extracted person saved to outputs/person_extracted_deep.jpg")
    
    return person_mask_refined


if __name__ == "__main__":
    test_deep_segmentation()
