# ocra

**ocra** — Python-библиотека с инструментами анализа фрагментов печатного текста (сканы, вырезки и т.п.).
Первый инструмент — **`OrientationPredictor`**: 

## Установка

```bash
pip install ocra
```

---

## OrientationPredictor 
Определяет ориентацию фрагмента как `HORZ` (горизонт) или `VERT` (вертикаль).
![Объяснение OrientationPredictor](./explaing_orient.py.png)
Пример:
```python
from ocra import OrientationPredictor

IMAGE_PATH = r"examples\hrk_463.png"

def main():
    model = OrientationPredictor()

    result = model.predict_path(IMAGE_PATH)
    label = "VERT" if result["pred"] == 1 else "HORZ"

    print("Path:", result["path"])
    print("Pred:", label, f"(class={result['pred']})")
    print(f"Prob VERT: {result['prob_vert']:.4f}")
    print(f"Prob HORZ: {result['prob_horz']:.4f}")
    print(f"Aspect (w/h): {result['aspect']:.4f}")

if __name__ == "__main__":
    main()
```

Пример вывода:

```
Path: examples\hrk_463.png
Pred: HORZ (class=0)
Prob VERT: 0.0001
Prob HORZ: 0.9999
Aspect (w/h): 6.5926
```
