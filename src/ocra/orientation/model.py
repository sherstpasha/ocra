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
        return int(out.shape[1])

    def forward(self, x: torch.Tensor, aspect: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        z = torch.cat([feats, aspect.view(-1, 1)], dim=1)
        return self.classifier(z)
    
    def export_onnx(self, path: str, input_size=(224, 224)):
        self.eval()
        dummy_image = torch.randn(1, 3, *input_size)
        dummy_aspect = torch.tensor([[1.0]])
        
        torch.onnx.export(
            self,
            (dummy_image, dummy_aspect),
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['image', 'aspect'],
            output_names=['logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'aspect': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
