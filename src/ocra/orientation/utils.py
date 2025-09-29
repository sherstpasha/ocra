import json
import os
import timm
from timm.data import resolve_data_config

def _get_data_cfg_compat(model_name: str):
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