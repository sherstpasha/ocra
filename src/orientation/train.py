import os, csv, json, random, logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import OrientationDataset
from model import OrientationModel
from utils import Config


# ------------------------- utils -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logger(exp_dir: str) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO); sh.setFormatter(fmt)
    logger.addHandler(sh)

    os.makedirs(exp_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(exp_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.INFO); fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler,
                    epoch: int, global_step: int, best_val_loss: float, best_val_acc: float,
                    extra: Dict[str, Any] | None = None):
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


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer and ckpt.get("optimizer"):
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and ckpt.get("scheduler"):
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def save_weights(path: str, model: nn.Module):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def collate_with_aspect(batch):
    xs, labels, paths, aspects = [], [], [], []
    for x, y, p, a in batch:
        xs.append(x); labels.append(y); paths.append(p); aspects.append(a)
    xs = torch.stack(xs, dim=0)
    labels = torch.stack(labels, dim=0)
    aspects = torch.stack(aspects, dim=0)
    return xs, labels, paths, aspects


def make_datasets_three_way(cfg: Config):
    exp_dir = getattr(cfg, "exp_dir", "exp1")
    os.makedirs(exp_dir, exist_ok=True)
    val_split_path  = os.path.join(exp_dir, "val_split.txt")
    test_split_path = os.path.join(exp_dir, "test_split.txt")

    base_train = OrientationDataset(cfg, is_train=True)
    n_total = len(base_train)
    if n_total == 0:
        raise RuntimeError("No images found for splitting")

    seed = int(getattr(cfg, "seed", 42))
    val_size  = int(getattr(cfg, "val_size", 0))
    test_size = int(getattr(cfg, "test_size", 0))
    val_frac  = float(getattr(cfg, "val_frac", 0.1))
    test_frac = float(getattr(cfg, "test_frac", 0.1))

    if val_size <= 0:
        val_size = max(1, int(round(n_total * val_frac)))
    if test_size <= 0:
        test_size = max(1, int(round(n_total * test_frac)))

    max_non_train = min(n_total - 1, val_size + test_size)
    if val_size + test_size > max_non_train:
        excess = val_size + test_size - max_non_train
        test_size = max(1, test_size - excess)

    if os.path.exists(val_split_path) and os.path.exists(test_split_path):
        with open(val_split_path,  "r", encoding="utf-8") as f:
            val_idxs  = [int(x) for x in f if x.strip()]
        with open(test_split_path, "r", encoding="utf-8") as f:
            test_idxs = [int(x) for x in f if x.strip()]
    else:
        rng = random.Random(seed)
        idxs = list(range(n_total))
        rng.shuffle(idxs)
        val_idxs  = sorted(idxs[:val_size])
        test_idxs = sorted(idxs[val_size: val_size + test_size])

        with open(val_split_path, "w", encoding="utf-8") as f:
            for i in val_idxs:  f.write(f"{i}\n")
        with open(test_split_path, "w", encoding="utf-8") as f:
            for i in test_idxs: f.write(f"{i}\n")

    # train — всё остальное
    val_set = set(val_idxs)
    test_set = set(test_idxs)
    train_idxs = [i for i in range(n_total) if i not in val_set and i not in test_set]

    ds_train = base_train
    ds_val   = OrientationDataset(cfg, is_train=False)
    ds_test  = OrientationDataset(cfg, is_train=False)

    train_dataset = Subset(ds_train, train_idxs)
    val_dataset   = Subset(ds_val,   val_idxs)
    test_dataset  = Subset(ds_test,  test_idxs)
    return train_dataset, val_dataset, test_dataset


# ------------------------- evaluation -------------------------
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad(), amp.autocast():
        for x, y, paths, aspect in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            aspect = aspect.to(device, non_blocking=True)

            logits = model(x, aspect)
            loss = criterion(logits, y)
            total_loss += float(loss.item())
            total_correct += int((logits.argmax(1) == y).sum().item())
            total_count += int(y.numel())
    avg_loss = total_loss / max(1, len(loader))
    acc = (total_correct / max(1, total_count)) if total_count > 0 else 0.0
    return avg_loss, acc


# ------------------------- main training -------------------------
def run_training(cfg_path: str, device_str: str = "cuda"):
    cfg = Config(cfg_path)
    set_seed(getattr(cfg, "seed", 42))

    exp_dir = getattr(cfg, "exp_dir", None)
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logger(exp_dir)
    logger.info("Start training (Orientation)")
    logger.info(f"Experiment dir: {exp_dir}")

    try:
        cfg.save()
        logger.info("Saved config to exp_dir/config.json")
    except Exception as e:
        logger.info(f"Config save skipped: {e}")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- hyperparams ---
    batch_size = int(getattr(cfg, "batch_size", 32))
    epochs = int(getattr(cfg, "epochs", 10))
    lr = float(getattr(cfg, "lr", 1e-3))
    optimizer_name = getattr(cfg, "optimizer", "AdamW")
    scheduler_name = getattr(cfg, "scheduler", "ReduceLROnPlateau")
    weight_decay = float(getattr(cfg, "weight_decay", 1e-4))
    momentum = float(getattr(cfg, "momentum", 0.9))
    num_workers = int(getattr(cfg, "num_workers", 4))
    save_every = int(getattr(cfg, "save_every", 1))
    resume_path = getattr(cfg, "resume_path", None)

    # --- data ---
    train_dataset, val_dataset, test_dataset = make_datasets_three_way(cfg)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_with_aspect, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_with_aspect, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_with_aspect, pin_memory=True
    )

    # --- model ---
    model = OrientationModel(cfg).to(device)
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # scheduler
    if scheduler_name == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=False, min_lr=1e-7)
    elif scheduler_name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name in ("None", None):
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    scaler = amp.GradScaler()

    # --- logs & paths ---
    log_dir = os.path.join(exp_dir, "logs"); os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    metrics_csv_path = os.path.join(exp_dir, "metrics_epoch.csv")
    if not os.path.exists(metrics_csv_path):
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])

    best_loss_path = os.path.join(exp_dir, "best_loss_ckpt.pth")
    best_acc_path  = os.path.join(exp_dir, "best_acc_ckpt.pth")
    last_path      = os.path.join(exp_dir, "last_ckpt.pth")
    best_loss_weights_path = os.path.join(exp_dir, "best_loss_weights.pth")
    best_acc_weights_path  = os.path.join(exp_dir, "best_acc_weights.pth")
    last_weights_path      = os.path.join(exp_dir, "last_weights.pth")

    # --- resume ---
    start_epoch, global_step = 1, 0
    best_val_loss, best_val_acc = float("inf"), -1.0
    if resume_path and os.path.isfile(resume_path):
        ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        best_val_acc = float(ckpt.get("best_val_acc", best_val_acc))
        logger.info(f"Resumed from: {resume_path} (epoch={start_epoch-1}, step={global_step})")

    # --- training loop ---
    for epoch in range(start_epoch, epochs + 1):
        # train
        model.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}", leave=False)
        for x, y, paths, aspect in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            aspect = aspect.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast():
                logits = model(x, aspect)            # (B,2)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_val = float(loss.item())
            total_train_loss += loss_val
            writer.add_scalar("Loss/train_step", loss_val, global_step)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], global_step)
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            global_step += 1

        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # validate
        avg_val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # tensorboard
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # CSV
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                epoch,
                f"{avg_train_loss:.6f}",
                f"{avg_val_loss:.6f}",
                f"{val_acc:.6f}",
                f"{optimizer.param_groups[0]['lr']:.6e}",
            ])

        msg = (f"Epoch {epoch:03d}/{epochs} | "
               f"train_loss={avg_train_loss:.4f} | val_loss={avg_val_loss:.4f} | "
               f"acc={val_acc:.4f} | lr={optimizer.param_groups[0]['lr']:.2e}")
        print(msg)
        logger.info(msg)

        # save "last" (по расписанию)
        if (epoch % save_every) == 0:
            save_checkpoint(last_path, model, optimizer, scheduler, scaler,
                            epoch, global_step, avg_val_loss, val_acc,
                            extra={"batch_size": batch_size, "lr": lr, "optimizer": optimizer_name,
                                   "scheduler": scheduler_name})
            save_weights(last_weights_path, model)

        # bests
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(best_loss_path, model, optimizer, scheduler, scaler,
                            epoch, global_step, best_val_loss, val_acc)
            save_weights(best_loss_weights_path, model)
            logger.info(f"New best val_loss: {best_val_loss:.4f} (epoch {epoch})")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(best_acc_path, model, optimizer, scheduler, scaler,
                            epoch, global_step, best_val_loss, best_val_acc)
            save_weights(best_acc_weights_path, model)
            logger.info(f"New best acc: {best_val_acc:.4f} (epoch {epoch})")

        # scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

    writer.close()
    logger.info("Training finished.")

    best_acc_ckpt = os.path.join(exp_dir, "best_acc_ckpt.pth")
    if os.path.isfile(best_acc_ckpt):
        logger.info("Loading best_acc_ckpt.pth for final test evaluation...")
        load_checkpoint(best_acc_ckpt, model, optimizer=None, scheduler=None, scaler=None)
    else:
        logger.info("best_acc_ckpt.pth not found, evaluating current weights on test set...")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info(f"TEST | loss={test_loss:.4f} | acc={test_acc:.4f}")

    with open(os.path.join(exp_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"test_loss": test_loss, "test_acc": test_acc}, f, indent=2)

    return {"val_acc": best_val_acc, "val_loss": best_val_loss,
            "test_acc": test_acc, "test_loss": test_loss, "exp_dir": exp_dir}


if __name__ == "__main__":
    run_training("config.json", device_str="cuda")
