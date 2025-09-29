import os
import json
from typing import Iterable, List, Dict, Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import importlib.resources as pkg_resources
import ocra.orientation as orientation_pkg

from .utils import Config, _get_data_cfg_compat
from .model import OrientationModel

__version__ = "0.1.4"


def _default_extensions() -> List[str]:
    return [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


class _PathsDataset(Dataset):
    def __init__(self, paths: List[str], model_name: str):
        self.paths = paths
        data_cfg = _get_data_cfg_compat(model_name)
        H, W = data_cfg["input_size"][1], data_cfg["input_size"][2]
        mean, std = data_cfg["mean"], data_cfg["std"]
        self.transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
        w, h = img.size
        aspect = float(w) / float(h) if h != 0 else 0.0
        x = self.transform(img)
        return x, torch.tensor(-1, dtype=torch.long), path, torch.tensor(aspect, dtype=torch.float32)


def _collate_with_aspect(batch):
    xs, ys, paths, aspects = [], [], [], []
    for x, y, p, a in batch:
        xs.append(x); ys.append(y); paths.append(p); aspects.append(a)
    return torch.stack(xs, 0), torch.stack(ys, 0), paths, torch.stack(aspects, 0)


class OrientationPredictor:
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        cfg: Union[str, Config, None] = None,
        batch_size: int = 64,
        num_workers: int = 4,
    ):
        if isinstance(cfg, Config):
            self.cfg = cfg
        elif isinstance(cfg, str):
            self.cfg = Config(cfg)
        else:
            with pkg_resources.as_file(
                pkg_resources.files(orientation_pkg).joinpath("config.json")
            ) as p:
                self.cfg = Config(str(p))

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)

        self.model = OrientationModel(self.cfg).to(self.device)
        self.model.eval()

        wp = (
            weights_path
            or getattr(self.cfg, "infer_weights_path", None)
            or self._find_default_weights()
        )
        if wp is not None and os.path.isfile(wp):
            state = torch.load(wp, map_location="cpu")
        else:
            with pkg_resources.as_file(
                pkg_resources.files(orientation_pkg).joinpath("best_acc_weights.pth")
            ) as p:
                state = torch.load(p, map_location="cpu")

        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            self.model.load_state_dict(state["model"], strict=True)
        else:
            self.model.load_state_dict(state, strict=True)

        self.extensions = [e.lower() for e in getattr(self.cfg, "extensions", _default_extensions())]

    def _find_default_weights(self) -> Optional[str]:
        exp_dir = getattr(self.cfg, "exp_dir", None)
        if not exp_dir:
            return None
        for name in ("best_acc_ckpt.pth", "best_acc_weights.pth", "last_ckpt.pth", "last_weights.pth"):
            p = os.path.join(exp_dir, name)
            if os.path.isfile(p):
                return p
        return None

    @torch.no_grad()
    def predict_path(self, path: str) -> Dict[str, Any]:
        res = self.predict_iterable([path])
        return res[0] if res else {}

    @torch.no_grad()
    def predict_folder(self, folder: str, recursive: bool = True) -> List[Dict[str, Any]]:
        return self._predict_paths(self._gather_paths(folder, recursive=recursive))

    @torch.no_grad()
    def predict_iterable(self, paths: Iterable[str]) -> List[Dict[str, Any]]:
        return self._predict_paths([p for p in paths if self._is_image_path(p)])

    def _is_image_path(self, path: str) -> bool:
        return any(path.lower().endswith(ext) for ext in self.extensions)

    def _gather_paths(self, root: str, recursive: bool = True) -> List[str]:
        if os.path.isfile(root):
            return [root] if self._is_image_path(root) else []
        files: List[str] = []
        if recursive:
            for dp, _, fns in os.walk(root):
                for fn in fns:
                    p = os.path.join(dp, fn)
                    if self._is_image_path(p):
                        files.append(p)
        else:
            for fn in os.listdir(root):
                p = os.path.join(root, fn)
                if os.path.isfile(p) and self._is_image_path(p):
                    files.append(p)
        files.sort()
        return files

    @torch.no_grad()
    def _predict_paths(self, paths: List[str]) -> List[Dict[str, Any]]:
        if not paths:
            return []
        ds = _PathsDataset(paths, self.cfg.model_name)
        loader = DataLoader(
            ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            collate_fn=_collate_with_aspect
        )
        outputs: List[Dict[str, Any]] = []
        for x, _, batch_paths, aspect in loader:
            x = x.to(self.device, non_blocking=True)
            aspect = aspect.to(self.device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                logits = self.model(x, aspect)
                probs_vert = F.softmax(logits, dim=1)[:, 1]
                preds = logits.argmax(1)
            for p, pred, pv, a in zip(batch_paths, preds.cpu().tolist(),
                                      probs_vert.cpu().tolist(), aspect.cpu().tolist()):
                outputs.append({
                    "path": p,
                    "pred": int(pred),
                    "prob_vert": float(pv),
                    "prob_horz": float(1.0 - pv),
                    "aspect": float(a),
                })
        return outputs

    @staticmethod
    def save_csv(results: List[Dict[str, Any]], out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        import csv
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "pred", "prob_vert", "prob_horz", "aspect"])
            for r in results:
                w.writerow([r["path"], r["pred"], f'{r["prob_vert"]:.6f}', f'{r["prob_horz"]:.6f}', f'{r["aspect"]:.6f}'])

    @staticmethod
    def save_json(results: List[Dict[str, Any]], out_path: str):
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
