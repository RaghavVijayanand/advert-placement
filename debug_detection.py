"""Debug: visualize what's being detected."""
import cv2
import numpy as np
from src.detectors import ProtectedContentDetector
from src.context_detector import ContextDetector

# Load image
image = cv2.imread('examples/living_room.jpg')

# Detect protected content
detector = ProtectedContentDetector()
detection = detector.detect(image)
protected_mask = detection['combined_mask']

# Detect screen
context = ContextDetector(verbose=False)
surfaces = context.detect_surfaces(image)
screen = [s for s in surfaces if s.type == 'screen'][0]

x, y, w, h = screen.x, screen.y, screen.w, screen.h

# Visualize
vis = image.copy()

# Draw screen in green
cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 3)

# Draw protected regions in red
vis_protected = cv2.cvtColor(protected_mask, cv2.COLOR_GRAY2BGR)
vis_protected[protected_mask > 0] = [0, 0, 255]
vis = cv2.addWeighted(vis, 0.7, vis_protected, 0.3, 0)

# Draw bboxes
for bbox in detection['bboxes']:
    bx, by, bw, bh, cls, conf = bbox
    cv2.rectangle(vis, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
    cv2.putText(vis, f"{cls}", (bx, by-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

cv2.imwrite('outputs/debug_detection.jpg', vis)
print("✓ Debug visualization saved to outputs/debug_detection.jpg")
print(f"  Screen: ({x}, {y}) {w}x{h}")
print(f"  Protected bboxes:")
for bbox in detection['bboxes']:
    bx, by, bw, bh, cls, conf = bbox
    print(f"    {cls}: ({bx}, {by}) {bw}x{bh}")
    
# Check overlap
protected_in_screen = protected_mask[y:y+h, x:x+w]
overlap_pixels = np.sum(protected_in_screen > 0)
print(f"  Protected pixels in screen region: {overlap_pixels}")
