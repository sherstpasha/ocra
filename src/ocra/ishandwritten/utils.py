

import os
import json
import sys
from typing import Dict, Any, Optional

try:
    from ..orientation.utils import Config, save_checkpoint, load_checkpoint
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    from src.ocra.orientation.utils import Config, save_checkpoint, load_checkpoint


class HandwrittenConfig(Config):
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            self._create_default_config()
        else:
            super().__init__(config_path)
        self._validate_handwritten_config()
    
    @classmethod
    def from_json(cls, config_path: str) -> 'HandwrittenConfig':
        return cls(config_path)
    
    def _create_default_config(self) -> None:
        config_path = os.path.join(os.path.dirname(__file__), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k, v in data.items():
                setattr(self, k, v)
        else:
            self.model_name = "mobilenetv3_small_050"
            self.num_classes = 2
            self.image_size = 224
            self.batch_size = 32
            self.learning_rate = 0.001
            self.num_epochs = 50
            self.hand_path = ""
            self.printed_path = ""
            self.train_split = 0.8
            self.val_split = 0.2
        
        if not hasattr(self, "exp_dir") or self.exp_dir is None:
            exp_idx = 1
            while os.path.exists(f"exp{exp_idx}"):
                exp_idx += 1
            self.exp_dir = f"exp{exp_idx}"
    
    def _validate_handwritten_config(self) -> None:
        required_fields = ['model_name', 'num_classes']
        
        for field in required_fields:
            if not hasattr(self, field):
                raise ValueError(f"Отсутствует обязательное поле конфигурации: {field}")
        
        if not hasattr(self, 'image_size'):
            if hasattr(self, 'input_size') and len(self.input_size) >= 2:
                self.image_size = self.input_size[-1]
            else:
                self.image_size = 224
        
        if hasattr(self, 'dataset'):
            if hasattr(self.dataset, 'hand_path'):
                self.hand_path = self.dataset['hand_path'] if isinstance(self.dataset, dict) else self.dataset.hand_path
            if hasattr(self.dataset, 'printed_path'):
                self.printed_path = self.dataset['printed_path'] if isinstance(self.dataset, dict) else self.dataset.printed_path

        if hasattr(self, 'hand_path') and self.hand_path and not os.path.exists(self.hand_path):
            pass
            
        if hasattr(self, 'printed_path') and self.printed_path and not os.path.exists(self.printed_path):
            pass
    



def get_class_names() -> Dict[int, str]:
    return {
        0: 'printed',
        1: 'handwritten'
    }


def get_class_colors() -> Dict[int, str]:
    """Возвращает цвета для визуализации классов."""
    return {
        0: '#3498db',
        1: '#e74c3c'
    }


def calculate_class_weights(hand_count: int, printed_count: int) -> Dict[int, float]:
    """
    Вычисляет веса классов для балансировки датасета.
    
    Args:
        hand_count: Количество рукописных примеров
        printed_count: Количество печатных примеров
        
    Returns:
        Словарь с весами для каждого класса
    """
    total = hand_count + printed_count
    
    if total == 0:
        return {0: 1.0, 1: 1.0}

    weight_printed = total / (2 * printed_count) if printed_count > 0 else 1.0
    weight_hand = total / (2 * hand_count) if hand_count > 0 else 1.0
    
    return {
        0: weight_printed,
        1: weight_hand
    }


def create_default_config(output_path: str, 
                         hand_path: str, 
                         printed_path: str) -> None:
    """
    Создает конфигурационный файл по умолчанию для handwritten модуля.
    
    Args:
        output_path: Путь для сохранения конфигурации
        hand_path: Путь к папке с рукописными изображениями  
        printed_path: Путь к папке с печатными изображениями
    """
    config = {
        "model_name": "efficientnet_b0",
        "num_classes": 2,
        "input_size": [3, 224, 224],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "dataset": {
            "hand_path": hand_path,
            "printed_path": printed_path,
            "train_split": 0.8,
            "val_split": 0.2
        },
        "training": {
            "batch_size": 32,
            "epochs": 50,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "patience": 10
        },
        "augmentation": {
            "horizontal_flip": 0.5,
            "vertical_flip": 0.0,
            "rotation": 15,
            "brightness": 0.1,
            "contrast": 0.1
        }
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"Конфигурация создана: {output_path}")


def validate_dataset_paths(config: HandwrittenConfig) -> Dict[str, Any]:
    """
    Проверяет пути к датасету и возвращает статистику.
    
    Args:
        config: Конфигурация с путями к данным
        
    Returns:
        Словарь со статистикой датасета
    """
    import glob
    
    stats = {
        'hand_path_exists': False,
        'printed_path_exists': False,
        'hand_count': 0,
        'printed_count': 0,
        'total_count': 0,
        'extensions_found': set()
    }
    
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
    

    if hasattr(config.dataset, 'hand_path') and os.path.exists(config.dataset.hand_path):
        stats['hand_path_exists'] = True
        for ext in extensions:
            files = glob.glob(os.path.join(config.dataset.hand_path, ext))
            files.extend(glob.glob(os.path.join(config.dataset.hand_path, ext.upper())))
            stats['hand_count'] += len(files)
            if files:
                stats['extensions_found'].add(ext.replace('*', ''))
    

    if hasattr(config.dataset, 'printed_path') and os.path.exists(config.dataset.printed_path):
        stats['printed_path_exists'] = True
        for ext in extensions:
            files = glob.glob(os.path.join(config.dataset.printed_path, ext))
            files.extend(glob.glob(os.path.join(config.dataset.printed_path, ext.upper())))
            stats['printed_count'] += len(files)
            if files:
                stats['extensions_found'].add(ext.replace('*', ''))
    
    stats['total_count'] = stats['hand_count'] + stats['printed_count']
    stats['extensions_found'] = list(stats['extensions_found'])
    
    return stats


def print_dataset_info(config: HandwrittenConfig) -> None:
    """Выводит информацию о датасете."""
    print("=== Информация о датасете ===")
    
    stats = validate_dataset_paths(config)
    
    print(f"Рукописные изображения:")
    print(f"  Путь: {config.dataset.hand_path}")
    print(f"  Существует: {stats['hand_path_exists']}")
    print(f"  Количество: {stats['hand_count']}")
    
    print(f"Печатные изображения:")
    print(f"  Путь: {config.dataset.printed_path}")  
    print(f"  Существует: {stats['printed_path_exists']}")
    print(f"  Количество: {stats['printed_count']}")
    
    print(f"Всего изображений: {stats['total_count']}")
    print(f"Найденные форматы: {', '.join(stats['extensions_found'])}")
    
    if stats['total_count'] > 0:
        class_weights = calculate_class_weights(stats['hand_count'], stats['printed_count'])
        print(f"Веса классов:")
        print(f"  Печатный: {class_weights[0]:.3f}")
        print(f"  Рукописный: {class_weights[1]:.3f}")
    
    print("=" * 30)



__all__ = [
    'HandwrittenConfig',
    'get_class_names', 
    'get_class_colors',
    'calculate_class_weights',
    'create_default_config',
    'validate_dataset_paths',
    'print_dataset_info',
    'save_checkpoint',
    'load_checkpoint'
]