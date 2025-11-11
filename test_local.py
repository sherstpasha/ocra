#!/usr/bin/env python
"""
Локальный тест перед коммитом
Проверяет основные функции пакета
"""
import sys
import subprocess
from pathlib import Path


def run_command(cmd, description):
    """Запустить команду и вернуть результат"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    print(f"Команда: {cmd}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ FAILED (exit code: {result.returncode})")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT (>60s)")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Запустить все локальные тесты"""
    print("\n" + "="*60)
    print("🚀 OCRA - Pre-commit Test Suite")
    print("="*60)
    
    tests = []
    
    # Проверка файлов
    print("\n📁 Checking required files...")
    required_files = [
        "pyproject.toml",
        "README.md",
        "TRAIN_README.md",
        "QUICKSTART.md",
        "src/ocra/__init__.py",
        "src/ocra/orientation/config.json",
        "src/ocra/ishandwritten/config.json",
    ]
    
    all_files_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - NOT FOUND")
            all_files_exist = False
    
    tests.append(("Required Files", all_files_exist))
    
    # Импорт базовых модулей
    tests.append((
        "Import ocra",
        run_command(
            'python -c "import ocra; print(f\'ocra {ocra.__version__}\')"',
            "Checking ocra import"
        )
    ))
    
    # Импорт компонентов
    tests.append((
        "Import OrientationPredictor",
        run_command(
            'python -c "from ocra import OrientationPredictor; print(\'OK\')"',
            "Checking OrientationPredictor import"
        )
    ))
    
    tests.append((
        "Import HandwrittenPredictor",
        run_command(
            'python -c "from ocra import HandwrittenPredictor; print(\'OK\')"',
            "Checking HandwrittenPredictor import"
        )
    ))
    
    # Проверка зависимостей
    tests.append((
        "Check numpy",
        run_command(
            'python -c "import numpy; print(f\'numpy {numpy.__version__}\')"',
            "Checking numpy"
        )
    ))
    
    tests.append((
        "Check Pillow",
        run_command(
            'python -c "from PIL import Image; print(\'Pillow OK\')"',
            "Checking Pillow"
        )
    ))
    
    tests.append((
        "Check OpenCV",
        run_command(
            'python -c "import cv2; print(f\'opencv {cv2.__version__}\')"',
            "Checking OpenCV"
        )
    ))
    
    tests.append((
        "Check ONNX Runtime",
        run_command(
            'python -c "import onnxruntime as ort; print(f\'onnxruntime {ort.__version__}\')"',
            "Checking ONNX Runtime"
        )
    ))
    
    # Проверка dev зависимостей (опционально)
    print("\n📦 Checking dev dependencies (optional)...")
    
    has_torch = run_command(
        'python -c "import torch; print(f\'PyTorch {torch.__version__}\')"',
        "Checking PyTorch"
    )
    
    if has_torch:
        run_command(
            'python -c "import timm; print(f\'timm {timm.__version__}\')"',
            "Checking timm"
        )
        
        run_command(
            'python -c "from torch.utils.tensorboard import SummaryWriter; print(\'tensorboard OK\')"',
            "Checking tensorboard"
        )
    
    # Проверка pyproject.toml
    tests.append((
        "Validate pyproject.toml",
        run_command(
            'python -c "import toml; toml.load(\'pyproject.toml\'); print(\'Valid\')"',
            "Validating pyproject.toml"
        )
    ))
    
    # Результаты
    print("\n" + "="*60)
    print("📊 TEST RESULTS")
    print("="*60)
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {name}")
    
    print(f"\n{'='*60}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to commit.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before committing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
