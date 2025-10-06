# ocra

**ocra** — Python-библиотека для анализа изображений текста (сканы, вырезки и т.п.).

Два основных инструмента:
- **`OrientationPredictor`** — определение ориентации (вертикальная/горизонтальная)
- **`HandwrittenPredictor`** — классификация типа текста (рукописный/печатный)

## Установка

```bash
# С CPU поддержкой
pip install ocra[cpu] 

# С GPU поддержкой (требует CUDA + cuDNN)
pip install ocra[gpu]
```

---

## Быстрый старт

```python
from src.ocra.orientation import OrientationPredictor
from src.ocra.ishandwritten import HandwrittenPredictor

def main():
    # Инициализация
    orient_pred = OrientationPredictor(verbose=0)
    text_pred = HandwrittenPredictor(verbose=False)
    
    image = "examples/hrk_463.png"
    
    # Определение ориентации
    orient_res = orient_pred.predict_single(image)
    orientation = "VERT" if orient_res['pred_class'] == 1 else "HORZ"
    
    # Определение типа текста
    text_res = text_pred.predict_single(image)
    text_type = text_res['prediction']  # 'handwritten' или 'printed'
    
    print(f"Ориентация: {orientation} ({orient_res['confidence']:.3f})")
    print(f"Тип текста: {text_type} ({text_res['confidence']:.3f})")

if __name__ == "__main__":
    main()
```

**Пример вывода:**
```
Ориентация: HORZ (1.000)
Тип текста: handwritten (0.982)
```

---

## OrientationPredictor 
Определяет ориентацию изображения как горизонтальную (HORZ) или вертикальную (VERT).

![Объяснение OrientationPredictor](./explaing_orient.py.png)

```python
from src.ocra.orientation import OrientationPredictor

predictor = OrientationPredictor()
result = predictor.predict_single("examples/hrk_463.png")

print(f"Класс: {result['pred_class']}")  # 0=HORZ, 1=VERT
print(f"Предсказание: {result['prediction']}")  # 'horizontal' или 'vertical'
print(f"Уверенность: {result['confidence']:.4f}")
```

## HandwrittenPredictor
Классифицирует тип текста на изображении: рукописный или печатный.

```python
from src.ocra.ishandwritten import HandwrittenPredictor

predictor = HandwrittenPredictor()
result = predictor.predict_single("examples/hrk_463.png")

print(f"Класс: {result['pred_class']}")     # 0=printed, 1=handwritten
print(f"Предсказание: {result['prediction']}")  # 'handwritten' или 'printed'
print(f"Уверенность: {result['confidence']:.4f}")
print(f"Высокая уверенность: {result['high_confidence']}")
```
