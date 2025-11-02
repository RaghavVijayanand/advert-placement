"""Show the result of TV ad placement."""
import cv2
import numpy as np

# Load original and result
original = cv2.imread('examples/living_room.jpg')
result = cv2.imread('outputs/living_room_with_ad.jpg')

if original is None or result is None:
    print("Error: Could not load images")
    exit(1)

# Create side-by-side comparison
h1, w1 = original.shape[:2]
h2, w2 = result.shape[:2]

# Make same height
if h1 != h2:
    scale = h1 / h2
    result = cv2.resize(result, (int(w2 * scale), h1))
    h2, w2 = result.shape[:2]

# Concatenate
comparison = np.hstack([original, result])

# Add labels
cv2.putText(comparison, "ORIGINAL", (50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
cv2.putText(comparison, "WITH AD ON TV", (w1 + 50, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

# Save comparison
cv2.imwrite('outputs/comparison.jpg', comparison)
print("✓ Comparison saved to outputs/comparison.jpg")
print(f"  Original size: {original.shape[:2]}")
print(f"  Result size: {result.shape[:2]}")

# Also show just the result
cv2.imwrite('outputs/final_result.jpg', result)
print("✓ Final result saved to outputs/final_result.jpg")
