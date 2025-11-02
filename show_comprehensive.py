"""
Compare different ad placement methods with visualizations.
"""
import cv2
import numpy as np

def create_comprehensive_comparison():
    """Create detailed comparison of all methods."""
    print("Creating comprehensive comparison...")
    
    # Load original image
    original = cv2.imread('examples/living_room.jpg')
    
    # Load different outputs
    outputs = {
        'Original': original,
        'Person Mask': cv2.imread('outputs/person_mask_deep.jpg'),
        'Depth Map': cv2.imread('outputs/depth_map.jpg'),
        'Depth Mask': cv2.imread('outputs/depth_mask.jpg'),
        'Final Mask': cv2.imread('outputs/debug_screen_mask_deep.jpg'),
        'Final Result': cv2.imread('outputs/precise_tv_placement.jpg'),
    }
    
    # Create 2x3 grid
    h, w = 512, 512  # Resize for display
    grid = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    
    titles = list(outputs.keys())
    images = list(outputs.values())
    
    for i, (title, img) in enumerate(zip(titles, images)):
        if img is None:
            print(f"Warning: Could not load {title}")
            continue
        
        # Resize
        resized = cv2.resize(img, (w, h))
        
        # Add title
        cv2.putText(resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2)
        cv2.putText(resized, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (0, 0, 0), 1)
        
        # Place in grid
        row = i // 3
        col = i % 3
        grid[row*h:(row+1)*h, col*w:(col+1)*w] = resized
    
    # Save
    cv2.imwrite('outputs/comprehensive_comparison.jpg', grid)
    print("✓ Comprehensive comparison saved to outputs/comprehensive_comparison.jpg")
    
    # Also create before/after
    before = original.copy()
    after = images[-1]  # Final result
    
    if after is not None:
        # Side by side
        H = max(before.shape[0], after.shape[0])
        W_total = before.shape[1] + after.shape[1]
        side_by_side = np.zeros((H, W_total, 3), dtype=np.uint8)
        
        side_by_side[0:before.shape[0], 0:before.shape[1]] = before
        side_by_side[0:after.shape[0], before.shape[1]:] = after
        
        # Add labels
        cv2.putText(side_by_side, "BEFORE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (255, 255, 255), 3)
        cv2.putText(side_by_side, "BEFORE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 0, 255), 2)
        
        cv2.putText(side_by_side, "AFTER", (before.shape[1] + 50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(side_by_side, "AFTER", (before.shape[1] + 50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        cv2.imwrite('outputs/before_after.jpg', side_by_side)
        print("✓ Before/After comparison saved to outputs/before_after.jpg")

if __name__ == "__main__":
    create_comprehensive_comparison()
