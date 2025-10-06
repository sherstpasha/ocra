import torch
import os
from model import HandwrittenClassifier, create_model
from utils import HandwrittenConfig


def export_to_onnx(model_path: str = "../../../checkpoints/handwritten/best_model.pth", 
                   config_path: str = "config.json", 
                   output_path: str = "handwritten_model.onnx"):
    """
    Экспортирует обученную модель handwritten в ONNX формат
    
    Args:
        model_path: Путь к checkpoint файлу
        config_path: Путь к конфигурации
        output_path: Выходной путь для ONNX модели
    """
    # Загружаем конфигурацию
    cfg = HandwrittenConfig(config_path)
    
    # Создаем модель
    model = create_model(cfg)
    
    # Загружаем веса
    if os.path.isfile(model_path):
        print(f"Загружаем модель из {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Извлекаем состояние модели из checkpoint
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict, strict=True)
        print(f"✓ Веса загружены успешно")
    else:
        raise FileNotFoundError(f"Checkpoint не найден: {model_path}")
    
    # Переводим в режим инференса
    model.eval()
    
    # Создаем dummy input для экспорта
    dummy_input = torch.randn(1, 3, 224, 224)
    
    print(f"Экспортируем в ONNX: {output_path}")
    
    # Экспортируем в ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    # Проверяем размеры файлов
    if os.path.exists(model_path):
        checkpoint_size = os.path.getsize(model_path) / (1024 * 1024)  # МБ
    else:
        checkpoint_size = 0
        
    if os.path.exists(output_path):
        onnx_size = os.path.getsize(output_path) / (1024 * 1024)  # МБ
        compression_ratio = checkpoint_size / onnx_size if onnx_size > 0 else 0
        
        print(f"✓ Экспорт завершен!")
        print(f"  Checkpoint (.pth): {checkpoint_size:.2f} МБ")
        print(f"  ONNX модель: {onnx_size:.2f} МБ")
        if compression_ratio > 1:
            print(f"  Сжатие: {compression_ratio:.1f}x")
    
    return output_path


if __name__ == "__main__":
    export_to_onnx()