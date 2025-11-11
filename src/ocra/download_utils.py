"""
Утилита для автоматической загрузки ONNX моделей
"""
import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Прогресс-бар для загрузки файлов"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_model(url: str, output_path: str, force: bool = False) -> str:
    """
    Загружает модель если она не существует
    
    Args:
        url: URL модели
        output_path: Путь куда сохранить
        force: Принудительно перезагрузить даже если файл существует
    
    Returns:
        Путь к загруженному файлу
    """
    output_path = Path(output_path)
    
    if output_path.exists() and not force:
        return str(output_path)
    
    # Создаем директорию если не существует
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading model from {url}...")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
            urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
        
        print(f"✓ Model downloaded: {output_path}")
        return str(output_path)
    
    except Exception as e:
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError(f"Failed to download model from {url}: {e}")


def get_model_path(model_name: str, cache_dir: str = None) -> str:
    """
    Получает путь к модели, загружая её при необходимости
    
    Args:
        model_name: Имя модели ('orientation' или 'handwritten')
        cache_dir: Директория для кеша (по умолчанию ~/.cache/ocra)
    
    Returns:
        Путь к файлу модели
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "ocra"
    else:
        cache_dir = Path(cache_dir)
    
    model_urls = {
        "orientation": "https://github.com/sherstpasha/ocra/releases/download/weights/orientation_model.onnx",
        "handwritten": "https://github.com/sherstpasha/ocra/releases/download/weights/handwritten_model.onnx"
    }
    
    if model_name not in model_urls:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_urls.keys())}")
    
    model_path = cache_dir / f"{model_name}_model.onnx"
    url = model_urls[model_name]
    
    return download_model(url, str(model_path))
