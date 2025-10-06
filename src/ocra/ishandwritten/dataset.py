import os
import glob
import sys
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from ..orientation.utils import Config
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    from src.ocra.orientation.utils import Config


class HandwrittenDataset(Dataset):
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int], 
                 transform=None,
                 config: Config = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.config = config
        
        assert len(image_paths) == len(labels), "Количество путей и меток должно совпадать"
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            
            return {
                'image': image,
                'label': torch.tensor(label, dtype=torch.long),
                'path': image_path,
                'filename': os.path.basename(image_path)
            }
            
        except Exception as e:
            print(f"Ошибка загрузки изображения {image_path}: {e}")
            return {
                'image': torch.zeros((3, 224, 224)),
                'label': torch.tensor(0, dtype=torch.long),
                'path': image_path,
                'filename': os.path.basename(image_path)
            }
    
    @classmethod
    def from_folders(cls, 
                    hand_folder: str, 
                    printed_folder: str,
                    transform=None,
                    config: Config = None) -> 'HandwrittenDataset':
        image_paths = []
        labels = []
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        
        printed_paths = []
        if os.path.exists(printed_folder):
            for ext in extensions:
                printed_paths.extend(glob.glob(os.path.join(printed_folder, ext), recursive=False))
                printed_paths.extend(glob.glob(os.path.join(printed_folder, ext.upper()), recursive=False))
            printed_paths = list(set(printed_paths))
            image_paths.extend(printed_paths)
            labels.extend([0] * len(printed_paths))
         
        if os.path.exists(hand_folder):
            hand_paths = []
            for ext in extensions:
                hand_paths.extend(glob.glob(os.path.join(hand_folder, ext), recursive=False))
                hand_paths.extend(glob.glob(os.path.join(hand_folder, ext.upper()), recursive=False))
            hand_paths = list(set(hand_paths))
            image_paths.extend(hand_paths)
            labels.extend([1] * len(hand_paths))
        
        return cls(image_paths, labels, transform, config)


def get_transforms(config: Config, mode: str = 'train') -> A.Compose:
    input_size = config.input_size[1:]
    mean = config.mean
    std = config.std
    
    if mode == 'train':
        augmentation_config = config.augmentation if hasattr(config, 'augmentation') else {}
        transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.HorizontalFlip(p=augmentation_config.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=augmentation_config.get('vertical_flip', 0.0)),
            A.Rotate(limit=augmentation_config.get('rotation', 15), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=augmentation_config.get('brightness', 0.1),
                contrast_limit=augmentation_config.get('contrast', 0.1),
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3)
            ], p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    return transform


def create_dataloaders(config: Config, 
                      test_size: float = 0.2, 
                      random_state: int = 42) -> Tuple[DataLoader, DataLoader]:
    full_dataset = HandwrittenDataset.from_folders(
        hand_folder=config.dataset['hand_path'],
        printed_folder=config.dataset['printed_path'],
        config=config
    )
    
    train_indices, val_indices = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        stratify=full_dataset.labels,
        random_state=random_state
    )
    
    train_transform = get_transforms(config, mode='train')
    val_transform = get_transforms(config, mode='val')
    
    train_paths = [full_dataset.image_paths[i] for i in train_indices]
    train_labels = [full_dataset.labels[i] for i in train_indices]
    train_dataset = HandwrittenDataset(train_paths, train_labels, train_transform, config)
    
    val_paths = [full_dataset.image_paths[i] for i in val_indices]
    val_labels = [full_dataset.labels[i] for i in val_indices]
    val_dataset = HandwrittenDataset(val_paths, val_labels, val_transform, config)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    return train_dataloader, val_dataloader