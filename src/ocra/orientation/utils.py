import json
import os
import random
from typing import Dict, Any

# Опциональные импорты для training
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import timm
    from timm.data import resolve_data_config
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    timm = None
    resolve_data_config = None


def _get_data_cfg_compat(model_name: str):
    """
    Получает конфигурацию данных для модели.
    Если timm недоступен, возвращает дефолтные значения.
    """
    if not TIMM_AVAILABLE:
        # Дефолтные значения для mobilenetv3_small_050
        return {
            "input_size": (3, 224, 224),
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.224, 0.225),
        }
    
    try:
        pre_cfg = timm.get_pretrained_cfg(model_name)
        if hasattr(pre_cfg, "to_dict"):
            pre_cfg = pre_cfg.to_dict()
        else:
            pre_cfg = dict(pre_cfg)
        data_cfg = resolve_data_config(model=None, pretrained_cfg=pre_cfg)
        return data_cfg
    except Exception:
        model = timm.create_model(model_name, pretrained=True)
        try:
            from timm.data import resolve_model_data_config
            data_cfg = resolve_model_data_config(model)
        except ImportError:
            data_cfg = resolve_data_config(model=model)
        return data_cfg
    
class Config:
    def __init__(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for k, v in data.items():
            setattr(self, k, v)

        if not hasattr(self, "exp_dir") or self.exp_dir is None:
            exp_idx = 1
            while os.path.exists(f"exp{exp_idx}"):
                exp_idx += 1
            self.exp_dir = f"exp{exp_idx}"

    def save(self, out_path: str | None = None):
        if out_path is None:
            out_path = os.path.join(self.exp_dir, "config.json")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)

    def __getitem__(self, key):
        return getattr(self, key)


def set_seed(seed: int = 42):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def save_checkpoint(path: str, model, optimizer, scheduler, scaler,
                    epoch: int, global_step: int, best_val_loss: float, best_val_acc: float,
                    extra: Dict[str, Any] | None = None):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "scaler": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "extra": extra or {},
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None, scaler=None):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def save_weights(path: str, model):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training. Install with: pip install torch")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)