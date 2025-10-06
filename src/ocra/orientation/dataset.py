from torch.utils.data import Dataset, ConcatDataset, Subset
from torchvision import transforms
import os, csv, random
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


def make_datasets_three_way(cfg):
    exp_dir = getattr(cfg, "exp_dir", "exp1")
    os.makedirs(exp_dir, exist_ok=True)
    val_split_path = os.path.join(exp_dir, "val_split.txt")
    test_split_path = os.path.join(exp_dir, "test_split.txt")

    base_train = OrientationDataset(cfg, is_train=True)
    n_total = len(base_train)
    if n_total == 0:
        raise RuntimeError("No images found for splitting")

    seed = int(getattr(cfg, "seed", 42))
    val_split = float(getattr(cfg, "val_split", 0.15))
    test_split = float(getattr(cfg, "test_split", 0.15))

    if os.path.exists(val_split_path) and os.path.exists(test_split_path):
        with open(val_split_path, "r") as f:
            val_indices = [int(x.strip()) for x in f.readlines() if x.strip()]
        with open(test_split_path, "r") as f:
            test_indices = [int(x.strip()) for x in f.readlines() if x.strip()]
        train_indices = [i for i in range(n_total) if i not in val_indices and i not in test_indices]
    else:
        indices = list(range(n_total))
        random.seed(seed)
        random.shuffle(indices)

        n_val = int(n_total * val_split)
        n_test = int(n_total * test_split)

        val_indices = indices[:n_val]
        test_indices = indices[n_val:n_val + n_test]
        train_indices = indices[n_val + n_test:]

        with open(val_split_path, "w") as f:
            f.write("\n".join(map(str, val_indices)))
        with open(test_split_path, "w") as f:
            f.write("\n".join(map(str, test_indices)))

    train_ds = Subset(base_train, train_indices)
    
    base_val = OrientationDataset(cfg, is_train=False)
    val_ds = Subset(base_val, val_indices)
    test_ds = Subset(base_val, test_indices)

    return train_ds, val_ds, test_ds