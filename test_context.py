"""Test context detection on living room image."""
import cv2
from src.context_detector import ContextDetector

# Load the living room image
img = cv2.imread('image.png')

if img is None:
    print("Error: Could not load image.png")
    exit(1)

print(f"Loaded image: {img.shape[1]}x{img.shape[0]}")

# Detect surfaces
detector = ContextDetector(verbose=True)
surfaces = detector.detect_surfaces(img)

# Print results
print(f"\n✓ Found {len(surfaces)} ad-friendly surfaces:")
for i, s in enumerate(surfaces[:10], 1):
    print(f"  {i}. {s.type}: ({s.x},{s.y}) {s.w}x{s.h} - confidence: {s.confidence:.2f}")

# Visualize
vis = detector.visualize_surfaces(img, surfaces)
cv2.imwrite('outputs/detected_surfaces.jpg', vis)
print(f"\n✓ Visualization saved to outputs/detected_surfaces.jpg")
