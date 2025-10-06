import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import make_datasets_three_way, collate_with_aspect
from model import OrientationModel
from utils import Config, set_seed, save_checkpoint, save_weights


def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for x, labels, _, aspects in tqdm(loader, desc="Training"):
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        aspects = aspects.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(x, aspects)
            loss = nn.CrossEntropyLoss()(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def validate_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, labels, _, aspects in tqdm(loader, desc="Validating"):
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            aspects = aspects.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x, aspects)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), correct / total


def train(cfg_path: str):
    cfg = Config(cfg_path)
    set_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_ds, val_ds, test_ds = make_datasets_three_way(cfg)
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_with_aspect, drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=collate_with_aspect
    )
    
    model = OrientationModel(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=cfg.patience)
    scaler = amp.GradScaler(enabled=device.type == "cuda")
    
    best_val_acc = 0.0
    best_val_loss = float("inf")
    
    print(f"Training for {cfg.epochs} epochs")
    
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss, val_acc = validate_epoch(model, val_loader, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_weights(os.path.join(cfg.exp_dir, "best_acc_weights.pth"), model)
            print(f"New best accuracy: {best_val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                os.path.join(cfg.exp_dir, "best_loss_ckpt.pth"),
                model, optimizer, scheduler, scaler,
                epoch, 0, best_val_loss, best_val_acc
            )
        
        save_checkpoint(
            os.path.join(cfg.exp_dir, "last_ckpt.pth"),
            model, optimizer, scheduler, scaler,
            epoch, 0, best_val_loss, best_val_acc
        )
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    train("src/ocra/orientation/config.json")