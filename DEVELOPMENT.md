# Руководство разработчика

## Установка для разработки

### Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone https://github.com/sherstpasha/Ocra.git
cd Ocra

# 2. Установите PyTorch (выберите версию для вашей CUDA)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# 3. Установите пакет в режиме разработки
pip install -e ".[dev,gpu]"
```

### Что установится?

**Базовые зависимости (inference):**
- `numpy` - работа с массивами
- `Pillow` - загрузка/обработка изображений
- `opencv-python` - компьютерное зрение
- `tqdm` - прогресс-бары

**GPU/CPU поддержка:**
- `onnxruntime-gpu` (или `onnxruntime` для CPU)

**Dev зависимости (training):**
- `timm` - библиотека моделей PyTorch
- `tensorboard` - визуализация обучения
- `scikit-learn` - утилиты ML
- `onnx` - экспорт моделей

**Важно:** PyTorch устанавливается отдельно!

## Структура проекта

```
ocra/
├── src/ocra/              # Исходный код
│   ├── __init__.py       # Публичный API
│   ├── orientation/      # Модуль определения ориентации
│   │   ├── config.json
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predictor.py
│   │   ├── prepare_dataset.py  # Создание CSV
│   │   └── orientation_model.onnx
│   └── ishandwritten/    # Модуль определения рукописного текста
│       ├── config.json
│       ├── dataset.py
│       ├── model.py
│       ├── train.py
│       ├── predictor.py
│       └── handwritten_model.onnx
├── examples/             # Примеры использования
├── pyproject.toml       # Конфигурация пакета
├── README.md
├── TRAIN_README.md      # Инструкции по обучению
├── QUICKSTART.md        # Быстрый старт
└── DEVELOPMENT.md       # Этот файл
```

## Workflow

### 1. Обучение новой модели

См. [QUICKSTART.md](QUICKSTART.md) для деталей.

```bash
# Подготовить конфиг
nano src/ocra/orientation/config.json

# Запустить обучение
python src/ocra/orientation/train.py
```

### 2. Экспорт в ONNX

```bash
python src/ocra/orientation/export_onnx.py
```

### 3. Тестирование inference

```python
from ocra import OrientationPredictor

pred = OrientationPredictor()
result = pred.predict_single("test_image.jpg")
print(result)
```

### 4. Запуск примеров

```bash
python examples/prepare_example.py --mode info
python simple_example.py
```

## Создание релиза

### 1. Обновите версию

Отредактируйте `pyproject.toml`:
```toml
version = "0.2.2"
```

### 2. Соберите пакет

```bash
pip install build
python -m build
```

### 3. Проверьте локально

```bash
pip install dist/ocra-0.2.2-py3-none-any.whl
```

### 4. Загрузите на PyPI

```bash
pip install twine
twine upload dist/*
```

## Тестирование

### Локальный тест перед коммитом

```bash
python test_local.py
```

Этот скрипт проверит:
- ✅ Наличие всех обязательных файлов
- ✅ Импорт основных модулей
- ✅ Наличие всех зависимостей
- ✅ Валидность pyproject.toml

### Проверка установки

```bash
# CPU версия
pip install -e ".[cpu,dev]"

# GPU версия
pip install -e ".[gpu,dev]"
```

### Проверка зависимостей

```python
import torch
import timm
import onnxruntime as ort

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"timm: {timm.__version__}")
print(f"ONNX Runtime: {ort.__version__}")
print(f"Providers: {ort.get_available_providers()}")
```

## Зависимости по модулям

### Inference only (пользователи)

```bash
pip install ocra[cpu]  # или ocra[gpu]
```

Включает:
- Core dependencies (numpy, Pillow, opencv-python, tqdm)
- onnxruntime или onnxruntime-gpu

### Development/Training (разработчики)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[dev,gpu]"
```

Включает всё выше плюс:
- timm
- tensorboard
- scikit-learn
- onnx

## FAQ

**Q: Почему PyTorch не в dependencies?**

A: PyTorch зависит от версии CUDA, которая индивидуальна для каждого пользователя. Автоматическая установка может поставить неправильную версию.

**Q: Как обновить зависимости?**

A: Отредактируйте `pyproject.toml`, затем:
```bash
pip install -e ".[dev,gpu]" --force-reinstall
```

**Q: Нужен ли requirements.txt?**

A: Нет, используйте `pyproject.toml`. Файл `requirements.txt` оставлен для обратной совместимости.

**Q: Как добавить новую модель?**

A: Скопируйте структуру `orientation/` или `ishandwritten/`, адаптируйте код, обновите `__init__.py`.

## Полезные команды

```bash
# Локальный тест перед коммитом
python test_local.py

# Переустановка в dev режиме
pip install -e ".[dev,gpu]" --force-reinstall

# Проверка установленного пакета
pip show ocra

# Список зависимостей
pip list | grep -E "torch|timm|onnx|numpy"

# Очистка билдов
rm -rf build/ dist/ *.egg-info

# Мониторинг обучения
tensorboard --logdir=exp_orientation_b0
```

## CI/CD

Проект использует GitHub Actions для автоматического тестирования.

📚 **См. документацию:** `.github/CI_CD.md`

Workflows:
- `test.yml` - Тестирование установки и примеров на разных ОС/Python
- `compatibility.yml` - Проверка совместимости с разными версиями зависимостей
- `quality.yml` - Линтинг, форматирование, проверка документации

Статус: [![Test](https://github.com/sherstpasha/Ocra/actions/workflows/test.yml/badge.svg)](https://github.com/sherstpasha/Ocra/actions/workflows/test.yml)

## Ссылки

- 📚 [PyTorch](https://pytorch.org/)
- 📚 [timm](https://github.com/huggingface/pytorch-image-models)
- 📚 [ONNX Runtime](https://onnxruntime.ai/)
- 📚 [Python Packaging](https://packaging.python.org/)
