import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from src.cnn_model import CNNModelBuilder
from src.utils import plot_training_history
import os

def run_cnn_pipeline():
    print("="*50)
    print("         STARTING DAY 2: CNN PIPELINE          ")
    print("="*50)

    # Step 1: Directories setup
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Step 2: Load Standard Dataset (CIFAR-10)
    print("\n[INFO] Loading CIFAR-10 Image Dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Normalize pixel values to range [0, 1]
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Limit samples for quick baseline training
    X_train_sub, y_train_sub = X_train[:5000], y_train[:5000]
    X_test_sub, y_test_sub = X_test[:1000], y_test[:1000]

    print(f"[INFO] Train shape: {X_train_sub.shape}, Test shape: {X_test_sub.shape}")

    # Step 3: Initialize and Build CNN
    cnn_builder = CNNModelBuilder(input_shape=(32, 32, 3), num_classes=10)
    model = cnn_builder.build_model()
    model.summary()

    # Step 4: Train CNN Model
    print("\n[INFO] Training CNN Model...")
    history = model.fit(
        X_train_sub, y_train_sub,
        epochs=10,
        batch_size=64,
        validation_data=(X_test_sub, y_test_sub),
        verbose=1
    )

    # Step 5: Evaluate Model
    test_loss, test_acc = model.evaluate(X_test_sub, y_test_sub, verbose=0)
    print(f"\n[RESULTS] Final Test Loss    : {test_loss:.4f}")
    print(f"[RESULTS] Final Test Accuracy: {test_acc * 100:.2f}%")

    # Step 6: Save Model Artifact
    model.save("models/cnn_model.keras")
    print("[INFO] CNN Model saved to: models/cnn_model.keras")

    print("\n==================================================")
    print("      CNN PIPELINE COMPLETED SUCCESSFULLY!        ")
    print("==================================================")

if __name__ == "__main__":
    run_cnn_pipeline()