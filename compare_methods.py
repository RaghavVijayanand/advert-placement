"""Compare the old vs new TV placement."""
import cv2
import numpy as np

# Load both results
old = cv2.imread('outputs/living_room_with_ad.jpg')
new = cv2.imread('outputs/precise_tv_placement.jpg')
original = cv2.imread('examples/living_room.jpg')

if old is None or new is None or original is None:
    print("Error loading images")
    exit(1)

# Create 3-way comparison
h, w = original.shape[:2]
comparison = np.hstack([original, old, new])

# Add labels
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(comparison, "ORIGINAL", (20, 40), font, 1, (0, 255, 0), 2)
cv2.putText(comparison, "OLD METHOD", (w + 20, 40), font, 1, (0, 0, 255), 2)
cv2.putText(comparison, "NEW METHOD", (2*w + 20, 40), font, 1, (0, 255, 0), 2)

cv2.imwrite('outputs/method_comparison.jpg', comparison)
print("✓ Comparison saved to outputs/method_comparison.jpg")
print("  Check if the person's head is properly preserved in NEW METHOD")
