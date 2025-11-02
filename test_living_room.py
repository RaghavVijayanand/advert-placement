"""Test context-aware ad placement on living room image."""
import cv2
from src.pipeline import AdPlacementPipeline

# Save the living room image to expected location
import shutil
shutil.copy('image.png', 'examples/living_room.jpg')

# Run analysis with context-aware detection
pipeline = AdPlacementPipeline(verbose=True)
result = pipeline.analyze('examples/living_room.jpg')

print("\n=== RESULTS ===")
print(f"Total candidates: {len(result['candidates'])}")
print("\nTop 5 candidates:")
for i, cand in enumerate(result['candidates'][:5], 1):
    print(f"  #{i}: ({cand.x}, {cand.y}) {cand.w}x{cand.h}")
    print(f"      Score: {cand.score:.3f}")
    print(f"      Reasons: {', '.join(cand.reasons)}")
    print()

# Save visualizations
cv2.imwrite('outputs/living_room_analysis.jpg', result['image'])
print("✓ Results saved to outputs/living_room_analysis.jpg")
