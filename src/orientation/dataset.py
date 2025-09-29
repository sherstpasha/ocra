from torch.utils.data import Dataset
from torchvision import transforms
import os, csv
from PIL import Image
import torch
from utils import _get_data_cfg_compat


class OrientationDataset(Dataset):
    def __init__(self, cfg, is_train=True):
        self.cfg = cfg
        self.is_train = is_train

        csv_path = getattr(cfg, "csv_path")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel = (row.get("file") or row.get("path") or "").strip()
                angle = int(row["angle"])

                if "label" in row and row["label"] != "" and row["label"] is not None:
                    label = int(row["label"])
                else:
                    label = 1 if angle % 180 != 0 else 0  # 90/270 -> 1, 0/180 -> 0
                rows.append((rel, angle, label))
        if not rows:
            raise RuntimeError(f"Empty CSV: {csv_path}")
        self.rows = rows

        data_cfg = _get_data_cfg_compat(cfg.model_name)
        H, W = data_cfg["input_size"][1], data_cfg["input_size"][2]
        mean, std = data_cfg["mean"], data_cfg["std"]
        fill_rgb = tuple(int(m * 255) for m in mean)

        self.mean, self.std = mean, std

        self.transform_train = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=2.0, translate=(0.01, 0.01), scale=(0.98, 1.02),
                    shear=0, fill=fill_rgb
                )
            ], p=0.25),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.05, hue=0.02)
            ], p=0.30),
            transforms.RandomAutocontrast(p=0.10),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


        self.root = getattr(cfg, "image_dir")

    @staticmethod
    def _rotate_90n_pil(img: Image.Image, angle: int) -> Image.Image:
        if angle % 360 == 0:
            return img
        k = (angle // 90) % 4
        if k == 1:
            return img.transpose(Image.ROTATE_90)
        if k == 2:
            return img.transpose(Image.ROTATE_180)
        if k == 3:
            return img.transpose(Image.ROTATE_270)
        return img

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rel, angle, label = self.rows[idx]

        path = rel
        if not os.path.isabs(path):
            path = os.path.join(self.root, rel)
        path = os.path.normpath(path)

        with Image.open(path) as im:
            img = im.convert("RGB")

        img = self._rotate_90n_pil(img, angle)

        w, h = img.size
        aspect = float(w) / float(h) if h != 0 else 0.0

        x = (self.transform_train if self.is_train else self.transform_test)(img)

        return x, torch.tensor(label, dtype=torch.long), path, torch.tensor(aspect, dtype=torch.float32)


def collate_with_aspect(batch):
    xs, labels, paths, aspects = [], [], [], []
    for x, y, p, a in batch:
        xs.append(x); labels.append(y); paths.append(p); aspects.append(a)
    xs = torch.stack(xs, dim=0)
    labels = torch.stack(labels, dim=0)
    aspects = torch.stack(aspects, dim=0)
    return xs, labels, paths, aspects