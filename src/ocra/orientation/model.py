import torch
import torch.nn as nn
import timm


class OrientationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        hidden_dim = getattr(cfg, "hidden_dim", 256)
        drop = getattr(cfg, "drop", 0.2)
        model_name = cfg.model_name

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features

        for p in self.backbone.parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if aspect.dim() == 1:
            aspect = aspect.unsqueeze(1)
        aspect = aspect.to(feats.dtype).to(feats.device)
        z = torch.cat([feats, aspect], dim=1)
        return self.classifier(z)