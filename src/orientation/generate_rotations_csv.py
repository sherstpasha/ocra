# gen_rotations.py
import os, csv, random
from typing import Iterable, List

# настрой под себя
ROOT_DIR = r"C:/Users/USER/Desktop/archive_25_09/dataset/val/img"
EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
SEED = 42
P_VERTICAL = 0.5  # вероятность вертикали (label=1)

VERT = [90, 270]  # label=1
HORZ = [0, 180]   # label=0

def list_images(root: str, exts: Iterable[str]) -> List[str]:
    exts = tuple(e.lower() for e in exts)
    files = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                # сохраним только имя файла (относительно ROOT_DIR)
                rel = os.path.relpath(os.path.join(dp, fn), root)
                files.append(rel.replace("\\", "/"))
    files.sort()
    return files

def main():
    rng = random.Random(SEED)
    files = list_images(ROOT_DIR, EXTENSIONS)
    if not files:
        raise RuntimeError(f"No images found in: {ROOT_DIR}")

    out_csv = os.path.join(ROOT_DIR, "rotations.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "angle", "label"])
        for rel_path in files:
            if rng.random() < P_VERTICAL:
                angle, label = rng.choice(VERT), 1
            else:
                angle, label = rng.choice(HORZ), 0
            w.writerow([rel_path, angle, label])

    print(f"Saved {len(files)} rows → {out_csv}")

if __name__ == "__main__":
    main()
