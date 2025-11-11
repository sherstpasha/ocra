"""
Утилита для создания rotations.csv из папок с изображениями.
Сканирует указанные директории и создает CSV файл с путями к изображениям.
"""
import os
import csv
from pathlib import Path
from typing import List


def create_rotations_csv(
    image_dirs: List[str],
    output_csv: str,
    extensions: List[str] = None
) -> int:
    """
    Создает rotations.csv из списка папок с изображениями.
    
    Args:
        image_dirs: Список путей к папкам с изображениями
        output_csv: Путь к выходному CSV файлу
        extensions: Список расширений файлов (по умолчанию .jpg, .jpeg, .png, .bmp, .tif, .tiff)
    
    Returns:
        Количество найденных изображений
    """
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    
    extensions = [ext.lower() for ext in extensions]
    
    image_files = []
    
    # Собираем все изображения из всех указанных папок
    for image_dir in image_dirs:
        image_dir = os.path.normpath(image_dir)
        if not os.path.exists(image_dir):
            print(f"Warning: Directory not found: {image_dir}")
            continue
        
        print(f"Scanning directory: {image_dir}")
        
        for root, _, files in os.walk(image_dir):
            for filename in files:
                ext = os.path.splitext(filename)[1].lower()
                if ext in extensions:
                    full_path = os.path.join(root, filename)
                    image_files.append(full_path)
    
    if not image_files:
        raise RuntimeError(f"No images found in the specified directories!")
    
    # Создаем директорию для CSV если не существует
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    
    # Записываем CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "angle", "label"])
        writer.writeheader()
        
        for img_path in sorted(image_files):
            # По умолчанию: angle=0 (без поворота), label=0 (горизонтальная ориентация)
            writer.writerow({
                "file": img_path,
                "angle": 0,
                "label": 0
            })
    
    print(f"\nCreated {output_csv}")
    print(f"Total images: {len(image_files)}")
    
    return len(image_files)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create rotations.csv from image directories"
    )
    parser.add_argument(
        "--image_dirs",
        nargs="+",
        required=True,
        help="List of directories containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rotations.csv",
        help="Output CSV file path (default: rotations.csv)"
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        help="Image file extensions to include"
    )
    
    args = parser.parse_args()
    
    create_rotations_csv(
        image_dirs=args.image_dirs,
        output_csv=args.output,
        extensions=args.extensions
    )
