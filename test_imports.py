#!/usr/bin/env python
"""
Тест импорта без PyTorch
"""
import sys

print("Testing imports without PyTorch...")

try:
    from ocra import OrientationPredictor, HandwrittenPredictor
    print("✅ Import successful!")
    print(f"OrientationPredictor: {OrientationPredictor}")
    print(f"HandwrittenPredictor: {HandwrittenPredictor}")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ All imports working!")
