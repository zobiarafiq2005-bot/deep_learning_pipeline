import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class ModelPredictor:
    """
    Production-ready Inference class for loading trained model artifacts
    and running real-time predictions.
    """
    def __init__(self, model_path: str, class_names: list = None):
        self.model_path = model_path
        self.class_names = class_names or [f"Class_{i}" for i in range(10)]
        self.model = self._load_model()

    def _load_model(self):
        print(f"[INFO] Loading saved model artifact from: {self.model_path}")
        return load_model(self.model_path)

    def predict_sample(self, input_sample: np.ndarray):
        """
        Runs inference on a single or batch input sample.
        Returns predicted class label, index, and confidence score.
        """
        if len(input_sample.shape) == 3:
            input_sample = np.expand_dims(input_sample, axis=0)  # Add batch dimension

        probabilities = self.model.predict(input_sample, verbose=0)
        predicted_idx = np.argmax(probabilities, axis=1)[0]
        confidence = float(np.max(probabilities, axis=1)[0])
        predicted_class = self.class_names[predicted_idx]

        return {
            "class_index": int(predicted_idx),
            "class_label": predicted_class,
            "confidence": round(confidence * 100, 2)
        }