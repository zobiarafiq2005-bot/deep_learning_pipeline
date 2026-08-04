import os
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from src.vgg16_model import VGG16TransferLearningBuilder
from src.utils import plot_training_history

def run_vgg16_pipeline():
    print("="*50)
    print("      STARTING DAY 3: VGG16 TRANSFER LEARNING    ")
    print("="*50)

    # Step 1: Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    # Step 2: Load CIFAR-10 Dataset
    print("\n[INFO] Loading CIFAR-10 Dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Preprocess inputs for VGG16 standard scaling
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # Subset data for fast baseline validation
    X_train_sub, y_train_sub = X_train[:5000], y_train[:5000]
    X_test_sub, y_test_sub = X_test[:1000], y_test[:1000]

    # Step 3: Initialize VGG16 Transfer Learning Model
    vgg_builder = VGG16TransferLearningBuilder(input_shape=(32, 32, 3), num_classes=10, freeze_base=True)
    model = vgg_builder.build_model()
    model.summary()

    # Step 4: Train Model
    print("\n[INFO] Training VGG16 Transfer Learning Pipeline...")
    history = model.fit(
        X_train_sub, y_train_sub,
        epochs=10,
        batch_size=64,
        validation_data=(X_test_sub, y_test_sub),
        verbose=1
    )

    # Step 5: Save Loss/Accuracy Plot
    plot_training_history(history, save_dir="outputs/figures", filename="vgg16_training_curves.png")

    # Step 6: Evaluate Model & Export Artifacts
    test_loss, test_acc = model.evaluate(X_test_sub, y_test_sub, verbose=0)
    print(f"\n[RESULTS] Final Test Loss    : {test_loss:.4f}")
    print(f"[RESULTS] Final Test Accuracy: {test_acc * 100:.2f}%")

    model.save("models/vgg16_model.keras")
    print("[INFO] VGG16 Model saved to: models/vgg16_model.keras")

    print("\n==================================================")
    print("    VGG16 PIPELINE COMPLETED SUCCESSFULLY!        ")
    print("==================================================")

if __name__ == "__main__":
    run_vgg16_pipeline()