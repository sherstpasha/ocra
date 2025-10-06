import torch
import torch.nn as nn
import timm
import os
import sys
from typing import Dict, Any

try:
    from ..orientation.utils import Config
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    from src.ocra.orientation.utils import Config


class HandwrittenClassifier(nn.Module):
    def __init__(self, model_name: str = "mobilenetv3_small_050", num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained,
            num_classes=0,
            global_pool=""
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_features = self.backbone(dummy_input)
            if len(backbone_features.shape) == 4:
                self.feature_dim = backbone_features.shape[1]
            else:
                self.feature_dim = backbone_features.shape[1]
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход модели.
        
        Args:
            x: Входные изображения [batch_size, 3, H, W]
            
        Returns:
            logits: Логиты классификации [batch_size, num_classes]
        """
        # Извлекаем признаки через backbone
        features = self.backbone(x)
        
        # Если признаки имеют пространственные измерения, применяем пулинг
        if len(features.shape) == 4:
            features = self.global_pool(features)
            features = features.flatten(1)
        
        # Классификация
        logits = self.classifier(features)
        
        return logits
    
    @classmethod
    def from_config(cls, config_path: str, **kwargs) -> 'HandwrittenClassifier':
        """Создание модели из конфигурационного файла."""
        config = Config(config_path)
        
        model_params = {
            'model_name': config.model_name,
            'num_classes': config.num_classes,
            'pretrained': kwargs.get('pretrained', True)
        }
        
        return cls(**model_params)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Dict = None, 
                       scheduler_state: Dict = None, **kwargs) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'optimizer_state_dict': optimizer_state,
            'scheduler_state_dict': scheduler_state,
            **kwargs
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = 'cpu') -> tuple:
        """Загрузка чекпойнта модели."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            model_name=checkpoint['model_name'],
            num_classes=checkpoint['num_classes'],
            pretrained=False
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def export_onnx(self, path: str, input_size: tuple = (1, 3, 224, 224), 
                   opset_version: int = 11) -> None:
        self.eval()
        dummy_input = torch.randn(input_size)
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX: {path}")


def create_model(config, **kwargs) -> HandwrittenClassifier:
    if isinstance(config, str):
        return HandwrittenClassifier.from_config(config, **kwargs)
    else:
        return HandwrittenClassifier(
            model_name=getattr(config, 'model_name', 'mobilenetv3_small_050'),
            num_classes=getattr(config, 'num_classes', 2),
            pretrained=kwargs.get('pretrained', True)
        )