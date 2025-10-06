import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2
from typing import Tuple, List, Union, Optional
import os

class HandwrittenPredictor:
    def __init__(
        self, 
        model_path: str = None, 
        confidence_threshold: float = 0.5, 
        verbose: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        
        module_dir = os.path.dirname(__file__)
        
        if model_path is None:
            model_path = os.path.join(module_dir, "handwritten_model.onnx")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found: {model_path}")
        
        self.model_path = model_path
        
        providers = ['CPUExecutionProvider']
        
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.session = ort.InferenceSession(model_path, providers=providers)
        
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        if verbose:
            print(f"Модель загружена: {os.path.basename(model_path)}")
            print(f"Провайдер: {self.session.get_providers()}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        image = image.resize((224, 224))
        image_array = np.array(image).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_array = (image_array - mean) / std
        
        image_array = np.transpose(image_array, (2, 0, 1))
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)
        
        return image_array
    
    def predict_single(self, image: Union[np.ndarray, Image.Image, str]) -> dict:
        processed_image = self.preprocess_image(image)
        
        outputs = self.session.run([self.output_name], {self.input_name: processed_image})
        logits = outputs[0][0]
        
        probabilities = self._softmax(logits)
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])
        
        class_names = {0: 'printed', 1: 'handwritten'}
        prediction_name = class_names[predicted_class]
        
        result = {
            'pred_class': predicted_class,
            'prediction': prediction_name,
            'class_name': prediction_name,
            'confidence': confidence,
            'prob_printed': float(probabilities[0]),
            'prob_handwritten': float(probabilities[1]),
            'probabilities': {
                'printed': float(probabilities[0]),
                'handwritten': float(probabilities[1])
            },
            'high_confidence': confidence >= self.confidence_threshold,
            'is_confident': confidence >= self.confidence_threshold
        }
        
        if isinstance(image, str):
            result['path'] = image
            result['filename'] = os.path.basename(image)
        
        return result
    
    def predict(self, images: List[Union[str, np.ndarray, Image.Image]], 
                filter_confident: bool = True) -> List[dict]:
        results = []
        for image in images:
            try:
                result = self.predict_single(image)
                if not filter_confident or result['is_confident']:
                    results.append(result)
            except Exception as e:
                print(f"Ошибка обработки изображения: {e}")
                continue
        return results
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)