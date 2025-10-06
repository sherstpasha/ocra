import os
import sys
import time
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

try:
    from .model import HandwrittenClassifier
    from .dataset import create_dataloaders
    from ..orientation.utils import Config
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    from src.ocra.ishandwritten.model import HandwrittenClassifier
    from src.ocra.ishandwritten.dataset import create_dataloaders
    from src.ocra.orientation.utils import Config


class HandwrittenTrainer:
    
    def __init__(self, config_path: str, device: str = None):
        self.config = Config(config_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = HandwrittenClassifier.from_config(config_path)
        self.model.to(self.device)
        
        self.train_loader, self.val_loader = create_dataloaders(
            self.config, 
            test_size=self.config.dataset['val_split']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.training['learning_rate'],
            weight_decay=self.config.training['weight_decay']
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='max',
            factor=0.5,
            patience=self.config.training['patience'] // 2,
            verbose=True
        )
        
        self.best_val_acc = 0.0
        self.best_model_path = None
        
        self.patience_counter = 0
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        running_loss = 0.0
        predictions = []
        targets = []
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = accuracy_score(targets, predictions)
        epoch_f1 = f1_score(targets, predictions, average='weighted')
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        
        running_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                predictions.extend(preds.cpu().numpy())
                targets.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{running_loss / (batch_idx + 1):.4f}'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = accuracy_score(targets, predictions)
        epoch_f1 = f1_score(targets, predictions, average='weighted')
        epoch_precision = precision_score(targets, predictions, average='weighted')
        epoch_recall = recall_score(targets, predictions, average='weighted')
        
        cm = confusion_matrix(targets, predictions)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'f1': epoch_f1,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'confusion_matrix': cm
        }
    
    def save_checkpoint(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False) -> None:
        checkpoint_dir = "checkpoints/handwritten"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "last_checkpoint.pth")
        self.model.save_checkpoint(
            checkpoint_path,
            epoch=epoch,
            optimizer_state=self.optimizer.state_dict(),
            scheduler_state=self.scheduler.state_dict(),
            val_metrics=val_metrics,
            config_path=self.config.config_path if hasattr(self.config, 'config_path') else None
        )
        
        if is_best:
            best_path = os.path.join(checkpoint_dir, "best_model.pth")
            self.model.save_checkpoint(
                best_path,
                epoch=epoch,
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                val_metrics=val_metrics,
                config_path=self.config.config_path if hasattr(self.config, 'config_path') else None
            )
            self.best_model_path = best_path
            print(f"Сохранен новый лучший чекпойнт: {best_path}")
    
    def train(self) -> None:
        print(f"Начинаем обучение на {self.config.training['epochs']} эпох")
        print("-" * 50)
        
        for epoch in range(self.config.training['epochs']):
            train_metrics = self.train_epoch(epoch)
            
            val_metrics = self.validate_epoch(epoch)
            

            
            self.scheduler.step(val_metrics['accuracy'])
            
            is_best = val_metrics['accuracy'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['accuracy']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            print(f"Epoch {epoch+1}/{self.config.training['epochs']}")
            print(f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
            print(f"Best Val Acc: {self.best_val_acc:.4f}")
            print(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")
            print("-" * 50)
            
            if self.patience_counter >= self.config.training['patience']:
                print(f"Early stopping после {epoch+1} эпох")
                break
        
        print("Обучение завершено!")
        print(f"Лучший результат валидации: {self.best_val_acc:.4f}")
        if self.best_model_path:
            print(f"Лучшая модель сохранена: {self.best_model_path}")



def main():
    import sys
    
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    
    if not os.path.exists(config_path):
        print(f"Файл конфигурации не найден: {config_path}")
        return
    
    trainer = HandwrittenTrainer(config_path)
    

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Тестовый режим - только инициализация")
        return
        
    trainer.train()


if __name__ == "__main__":
    main()