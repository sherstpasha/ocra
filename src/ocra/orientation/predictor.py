import os
import glob
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

from .utils import Config, _get_data_cfg_compat


class OrientationPredictor:
    def __init__(
        self, 
        model_path: str = None, 
        config_path: str = None,
        device: str = "auto",
        threshold: float = 0.5,
        verbose: int = 0
    ):
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")
        
        module_dir = os.path.dirname(__file__)
        
        if model_path is None:
            model_path = os.path.join(module_dir, "orientation_model.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        if config_path is None:
            config_path = os.path.join(module_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config not found: {config_path}")
        
        self.threshold = threshold
        self.verbose = verbose
        self.cfg = Config(config_path)
        
        providers = self._get_providers(device)
        
        # Создаем session с оптимальными настройками
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3 if not self.verbose else 0  # Подробные логи только в verbose режиме
        
        # Настройка провайдеров для оптимальной производительности
        final_providers = []
        for provider in providers:
            if provider == "CUDAExecutionProvider":
                provider_options = {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB limit
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                }
                final_providers.append(('CUDAExecutionProvider', provider_options))
            elif provider == "TensorrtExecutionProvider":
                provider_options = {
                    'device_id': 0,
                    'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB
                }
                final_providers.append(('TensorrtExecutionProvider', provider_options))
            else:
                final_providers.append(provider)
        
        try:
            self.session = ort.InferenceSession(model_path, providers=final_providers, sess_options=session_options)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to create session with GPU providers: {e}")
                print("Falling back to CPU...")
            # Fallback to CPU only
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=session_options)
        
        data_cfg = _get_data_cfg_compat(self.cfg.model_name)
        self.input_size = (data_cfg["input_size"][1], data_cfg["input_size"][2])
        self.mean = data_cfg["mean"]
        self.std = data_cfg["std"]
        
        self.transform = transforms.Compose([
            transforms.Resize(self.input_size, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        if self.verbose:
            print(f"OrientationPredictor initialized")
            print(f"Model: {os.path.basename(model_path)}")
            print(f"Available providers: {ort.get_available_providers()}")
            print(f"Active providers: {self.session.get_providers()}")
            print(f"Threshold: {threshold}")
        else:
            active_provider = self.session.get_providers()[0]
            device_info = "GPU" if "CUDA" in active_provider or "Tensorrt" in active_provider else "CPU"
            print(f"OCRA OrientationPredictor loaded ({device_info})")
    
    def _get_providers(self, device: str) -> List[str]:
        available = ort.get_available_providers()
        
        if device == "auto":
            # Проверяем доступность GPU провайдеров в порядке приоритета
            if "CUDAExecutionProvider" in available:
                if self.verbose:
                    print("Using CUDA GPU acceleration")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "TensorrtExecutionProvider" in available:
                if self.verbose:
                    print("Using TensorRT GPU acceleration")
                return ["TensorrtExecutionProvider", "CPUExecutionProvider"]
            else:
                if self.verbose:
                    print("No GPU providers available, using CPU")
                return ["CPUExecutionProvider"]
        elif device == "cuda" or device == "gpu":
            if "CUDAExecutionProvider" in available:
                if self.verbose:
                    print("Using CUDA GPU acceleration")
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "TensorrtExecutionProvider" in available:
                if self.verbose:
                    print("Using TensorRT GPU acceleration")  
                return ["TensorrtExecutionProvider", "CPUExecutionProvider"]
            else:
                print("Warning: GPU requested but no GPU providers available. Using CPU.")
                return ["CPUExecutionProvider"]
        else:
            if self.verbose:
                print("Using CPU")
            return ["CPUExecutionProvider"]
    
    def _is_image_file(self, path: str) -> bool:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp']
        return any(path.lower().endswith(ext) for ext in extensions)
    
    def _collect_image_paths(self, path: str) -> List[str]:
        if os.path.isfile(path):
            return [path] if self._is_image_file(path) else []
        elif os.path.isdir(path):
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff', '*.webp']:
                images.extend(glob.glob(os.path.join(path, ext), recursive=False))
                images.extend(glob.glob(os.path.join(path, ext.upper()), recursive=False))
            return sorted(list(set(images)))
        else:
            return []
    
    def _preprocess_image(self, image_path: str) -> tuple:
        try:
            with Image.open(image_path) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                aspect = float(w) / float(h) if h > 0 else 1.0
                
                # Применяем трансформации
                img_tensor = self.transform(img_rgb).unsqueeze(0)  # [1, 3, H, W]
                aspect_tensor = np.array([[aspect]], dtype=np.float32)  # [1, 1]
                
                return img_tensor.numpy(), aspect_tensor, aspect
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, None, None
    
    def _predict_batch(self, image_tensors: np.ndarray, aspect_tensors: np.ndarray) -> np.ndarray:
        inputs = {
            'image': image_tensors,
            'aspect': aspect_tensors
        }
        outputs = self.session.run(['logits'], inputs)
        return outputs[0]
    
    def predict(self, path: str, batch_size: int = 32) -> Dict[str, Any]:
        image_paths = self._collect_image_paths(path)
        
        if not image_paths:
            return {'results': [], 'summary': {'total': 0, 'processed': 0, 'vertical': 0, 'horizontal': 0, 'low_confidence': 0}}
        
        # Диагностика провайдера
        if self.verbose:
            print(f"Session providers in use: {self.session.get_providers()}")
            # Попробуем получить информацию о GPU
            try:
                import onnxruntime as ort_test
                print(f"ONNX Runtime build info: {ort_test.get_build_info()}")
            except:
                pass
        
        results = []
        batch_images = []
        batch_aspects = []
        batch_paths = []
        batch_original_aspects = []
        
        iterator = tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images") if self.verbose else enumerate(image_paths)
        
        for i, img_path in iterator:
            img_tensor, aspect_tensor, original_aspect = self._preprocess_image(img_path)
            
            if img_tensor is None:
                continue
                
            batch_images.append(img_tensor)
            batch_aspects.append(aspect_tensor)
            batch_paths.append(img_path)
            batch_original_aspects.append(original_aspect)
            
            if len(batch_images) == batch_size or i == len(image_paths) - 1:
                if batch_images:
                    batch_img_array = np.vstack(batch_images)
                    batch_aspect_array = np.vstack(batch_aspects)
                    
                    logits = self._predict_batch(batch_img_array, batch_aspect_array)
                    for j, (logits_single, img_path, original_aspect) in enumerate(
                        zip(logits, batch_paths, batch_original_aspects)):
                        
                        exp_logits = np.exp(logits_single - np.max(logits_single))
                        probs = exp_logits / np.sum(exp_logits)
                        
                        pred = int(np.argmax(logits_single))
                        prob_vert = float(probs[1])
                        prob_horz = float(probs[0])
                        confidence = max(prob_vert, prob_horz)
                        
                        result = {
                            'path': img_path,
                            'filename': os.path.basename(img_path),
                            'prediction': 'vertical' if pred == 1 else 'horizontal',
                            'pred_class': pred,
                            'prob_vertical': prob_vert,
                            'prob_horizontal': prob_horz,
                            'confidence': confidence,
                            'aspect_ratio': original_aspect,
                            'high_confidence': confidence >= self.threshold
                        }
                        results.append(result)
                
                batch_images.clear()
                batch_aspects.clear() 
                batch_paths.clear()
                batch_original_aspects.clear()
        
        total = len(image_paths)
        processed = len(results)
        vertical = sum(1 for r in results if r['pred_class'] == 1)
        horizontal = processed - vertical
        low_confidence = sum(1 for r in results if not r['high_confidence'])
        
        summary = {
            'total': total,
            'processed': processed,
            'vertical': vertical,
            'horizontal': horizontal,
            'low_confidence': low_confidence,
            'avg_confidence': np.mean([r['confidence'] for r in results]) if results else 0.0
        }
        
        return {
            'results': results,
            'summary': summary
        }
    
    def predict_single(self, image_path: str) -> Dict[str, Any]:
        result = self.predict(image_path)
        return result['results'][0] if result['results'] else None
    
    def filter_by_confidence(self, results: Dict[str, Any], min_confidence: float = None) -> Dict[str, Any]:
        if min_confidence is None:
            min_confidence = self.threshold
            
        filtered_results = [r for r in results['results'] if r['confidence'] >= min_confidence]
        
        processed = len(filtered_results)
        vertical = sum(1 for r in filtered_results if r['pred_class'] == 1)
        horizontal = processed - vertical
        
        summary = results['summary'].copy()
        summary.update({
            'filtered_count': processed,
            'filtered_vertical': vertical,
            'filtered_horizontal': horizontal,
            'filter_threshold': min_confidence
        })
        
        return {
            'results': filtered_results,
            'summary': summary
        }