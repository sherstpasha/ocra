import torch
import torch.nn as nn
import timm

class OrientationModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        hidden_dim = cfg.hidden_dim
        drop = cfg.drop
        model_name = cfg.model_name

        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")

        feat_dim = self._infer_feat_dim()

        for p in self.backbone.parameters():
            p.requires_grad = True

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, 2),
        )

    @torch.no_grad()
    def _infer_feat_dim(self) -> int:
        self.backbone.eval()

        dummy = torch.zeros(1, 3, 224, 224)
        out = self.backbone(dummy)

        if isinstance(out, (list, tuple)):
            out = out[-1]
        if out.ndim == 4:
            out = out.mean(dim=(2, 3))
        return int(out.shape[1])

    def forward(self, x: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]
        if feats.ndim == 4:
            feats = feats.mean(dim=(2, 3))

        if aspect.dim() == 1:
            aspect = aspect.unsqueeze(1)
        aspect = aspect.to(feats.dtype).to(feats.device)

        z = torch.cat([feats, aspect], dim=1)
        return self.classifier(z)
