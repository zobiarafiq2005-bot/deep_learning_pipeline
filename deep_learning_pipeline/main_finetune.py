import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from src.utils import plot_training_history, print_evaluation_report


def run_finetuning_pipeline():
    print("=" * 50)
    print("   STARTING DAY 4: VGG16 FINE-TUNING PIPELINE   ")
    print("=" * 50)

    # Step 1: Load CIFAR-10 Dataset
    print("\n[INFO] Loading Dataset...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    X_train_sub, y_train_sub = X_train[:5000], y_train[:5000]
    X_test_sub, y_test_sub = X_test[:1000], y_test[:1000]

    # Step 2: Load Previously Saved Day 3 Model
    model_path = "models/vgg16_model.keras"
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file '{model_path}' not found! Run main_vgg16.py first.")
        return

    print(f"\n[INFO] Loading saved VGG16 model from {model_path}...")
    model = load_model(model_path)

    # Step 3: Unfreeze top VGG16 layers for fine-tuning
    print("[INFO] Unfreezing top VGG16 convolutional layers...")

    # Access inner VGG16 layer
    vgg_base = None
    for layer in model.layers:
        if 'vgg16' in layer.name.lower():
            vgg_base = layer
            break

    if vgg_base:
        vgg_base.trainable = True
        # Freeze all layers except the last block (block5)
        for layer in vgg_base.layers[:-4]:
            layer.trainable = False

    # Re-compile with lower learning rate to preserve pre-trained features
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Step 4: Fine-tune training
    print("\n[INFO] Fine-Tuning Model (Epochs 1-5)...")
    history = model.fit(
        X_train_sub, y_train_sub,
        epochs=5,
        batch_size=64,
        validation_data=(X_test_sub, y_test_sub),
        verbose=1
    )

    # Step 5: Save Plots & Model
    plot_training_history(history, save_dir="outputs/figures", filename="vgg16_finetuned_curves.png")

    # Step 6: Full Evaluation
    test_loss, test_acc = model.evaluate(X_test_sub, y_test_sub, verbose=0)
    print(f"\n[RESULTS] Fine-Tuned Test Loss    : {test_loss:.4f}")
    print(f"[RESULTS] Fine-Tuned Test Accuracy: {test_acc * 100:.2f}%")

    y_pred_probs = model.predict(X_test_sub)
    y_pred = np.argmax(y_pred_probs, axis=1)

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print_evaluation_report(y_test_sub.flatten(), y_pred, target_names=cifar10_classes)

    model.save("models/vgg16_finetuned_model.keras")
    print("[INFO] Fine-tuned model saved to: models/vgg16_finetuned_model.keras")

    print("\n==================================================")
    print("   FINE-TUNING PIPELINE COMPLETED SUCCESSFULLY!   ")
    print("==================================================")


if __name__ == "__main__":
    run_finetuning_pipeline()