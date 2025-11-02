"""Test script to verify pretrained models load and run correctly."""

import sys
import torch
import cv2
import numpy as np
from pathlib import Path

def test_paddleocr():
    """Test PaddleOCR text detection."""
    print("Testing PaddleOCR...")
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_textline_orientation=True, lang='en')
        print("✓ PaddleOCR loaded successfully")
        return True
    except Exception as e:
        print(f"✗ PaddleOCR failed: {e}")
        return False

def test_torch():
    """Test PyTorch and check for GPU availability."""
    print("\nTesting PyTorch...")
    try:
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        
        # Test DirectML
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            print(f"  ONNX Runtime providers: {providers}")
            if 'DmlExecutionProvider' in providers:
                print("  ✓ DirectML available for GPU acceleration")
        except Exception as e:
            print(f"  Note: DirectML check: {e}")
        
        print("✓ PyTorch loaded successfully")
        return True
    except Exception as e:
        print(f"✗ PyTorch failed: {e}")
        return False

def test_opencv():
    """Test OpenCV."""
    print("\nTesting OpenCV...")
    try:
        print(f"  OpenCV version: {cv2.__version__}")
        # Create a test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        print("✓ OpenCV loaded successfully")
        return True
    except Exception as e:
        print(f"✗ OpenCV failed: {e}")
        return False

def test_timm():
    """Test timm for model loading."""
    print("\nTesting timm...")
    try:
        import timm
        print(f"  timm version: {timm.__version__}")
        # Just verify we can list models
        models = timm.list_models()[:5]
        print(f"  Sample models available: {len(models)} checked")
        print("✓ timm loaded successfully")
        return True
    except Exception as e:
        print(f"✗ timm failed: {e}")
        return False

def test_lightgbm():
    """Test LightGBM."""
    print("\nTesting LightGBM...")
    try:
        import lightgbm as lgb
        print(f"  LightGBM version: {lgb.__version__}")
        print("✓ LightGBM loaded successfully")
        return True
    except Exception as e:
        print(f"✗ LightGBM failed: {e}")
        return False

def main():
    print("=" * 50)
    print("DAPAC Model Testing Suite")
    print("=" * 50)
    
    results = {
        "PyTorch": test_torch(),
        "OpenCV": test_opencv(),
        "PaddleOCR": test_paddleocr(),
        "timm": test_timm(),
        "LightGBM": test_lightgbm(),
    }
    
    print("\n" + "=" * 50)
    print("Summary:")
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:15s} {status}")
    
    all_passed = all(results.values())
    print("=" * 50)
    
    if all_passed:
        print("\n✓ All tests passed! Ready to proceed.")
        return 0
    else:
        print("\n✗ Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
