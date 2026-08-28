import os
import numpy as np
from tensorflow.keras.datasets import cifar10
from src.predictor import ModelPredictor


def run_inference_pipeline():
    print("=" * 50)
    print("   STARTING DAY 5: INFERENCE & PREDICTION PIPELINE   ")
    print("=" * 50)

    model_path = "models/vgg16_finetuned_model.keras"

    # Fallback to standard VGG16 model if fine-tuned model doesn't exist
    if not os.path.exists(model_path):
        model_path = "models/vgg16_model.keras"

    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    # Initialize Predictor Engine
    predictor = ModelPredictor(model_path=model_path, class_names=cifar10_classes)

    # Load test dataset sample to simulate real-world unseen inputs
    print("\n[INFO] Simulating real-time inference on unseen test samples...")
    _, (X_test, y_test) = cifar10.load_data()
    X_test_normalized = X_test.astype('float32') / 255.0

    # Pick 5 random test samples
    sample_indices = np.random.choice(len(X_test_normalized), size=5, replace=False)

    print("\n" + "-" * 50)
    print(f"{'Sample #':<10} | {'Actual Class':<15} | {'Predicted Class':<15} | {'Confidence':<10}")
    print("-" * 50)

    for i, idx in enumerate(sample_indices, 1):
        sample_img = X_test_normalized[idx]
        actual_label = cifar10_classes[y_test[idx][0]]

        # Run Real-time Inference
        result = predictor.predict_sample(sample_img)

        print(f"{i:<10} | {actual_label:<15} | {result['class_label']:<15} | {result['confidence']}%")

    print("-" * 50)
    print("\n==================================================")
    print("   INFERENCE PIPELINE COMPLETED SUCCESSFULLY!    ")
    print("==================================================")


if __name__ == "__main__":
    run_inference_pipeline()