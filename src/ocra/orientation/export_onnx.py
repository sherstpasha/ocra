import torch
import os
from model import OrientationModel
from utils import Config


def export_to_onnx(model_path: str = "best_acc_weights.pth", 
                   config_path: str = "config.json", 
                   output_path: str = "orientation_model.onnx"):
    cfg = Config(config_path)
    model = OrientationModel(cfg)
    
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
    else:
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    model.eval()
    
    dummy_image = torch.randn(1, 3, 224, 224)
    dummy_aspect = torch.tensor([[1.0]], dtype=torch.float32)
    
    torch.onnx.export(
        model,
        (dummy_image, dummy_aspect),
        output_path,
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
    
    print(f"Exported to {output_path}")
    return output_path


if __name__ == "__main__":
    export_to_onnx()