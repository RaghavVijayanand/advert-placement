"""Demo: Place ad on the TV screen in living room."""
import cv2
import numpy as np
from src.pipeline import AdPlacementPipeline

# Create a sample ad (simple colored rectangle with text)
ad = np.ones((400, 600, 3), dtype=np.uint8)
ad[:] = (50, 50, 150)  # Dark red background

# Add text
cv2.putText(ad, "NEW PRODUCT", (100, 150), 
            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.putText(ad, "AVAILABLE NOW", (120, 280), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

# Save ad
cv2.imwrite('examples/sample_ad.png', ad)

# Place ad on living room TV
pipeline = AdPlacementPipeline(verbose=True)

print("\n=== ANALYZING LIVING ROOM ===")
analysis = pipeline.analyze('examples/living_room.jpg')

# Get top candidate (should be the TV!)
top_candidate = analysis['candidates'][0]
print(f"\n✓ Top candidate detected:")
print(f"  Type: {', '.join([r for r in top_candidate.reasons if 'context_' in r])}")
print(f"  Position: ({top_candidate.x}, {top_candidate.y})")
print(f"  Size: {top_candidate.w}x{top_candidate.h}")
print(f"  Score: {top_candidate.score:.3f}")

print("\n=== COMPOSITING AD ON TV SCREEN ===")
result = pipeline.compose(
    analysis['image'],
    'examples/sample_ad.png',
    top_candidate,
    analysis['protected_mask']
)

if result['valid']:
    print("\n✓ SUCCESS! Ad placed on TV screen")
    cv2.imwrite('outputs/living_room_with_ad.jpg', result['composite'])
    print(f"✓ Saved to: outputs/living_room_with_ad.jpg")
else:
    print("\n✗ Composition failed")
    if result['warnings']:
        for warning in result['warnings']:
            print(f"  Warning: {warning}")
