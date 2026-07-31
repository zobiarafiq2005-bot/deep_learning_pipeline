import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


class PlotUtils:
    """Helper class to save evaluation plots and reports."""

    @staticmethod
    def plot_training_history(history, save_dir: str = "outputs/figures"):
        """Plots Training vs Validation Loss & Accuracy."""
        os.makedirs(save_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss Plot
        ax1.plot(history.history['loss'], label='Train Loss', color='blue')
        ax1.plot(history.history['val_loss'], label='Val Loss', color='orange')
        ax1.set_title('ANN Loss Curve')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Accuracy Plot
        ax2.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='orange')
        ax2.set_title('ANN Accuracy Curve')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        plt.tight_layout()
        path = os.path.join(save_dir, 'ann_training_curves.png')
        plt.savefig(path)
        plt.close()
        print(f"[INFO] Training history saved to {path}")

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_dir: str = "outputs/figures"):
        """Plots Confusion Matrix heatmap."""
        os.makedirs(save_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('ANN Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        path = os.path.join(save_dir, 'ann_confusion_matrix.png')
        plt.savefig(path)
        plt.close()
        print(f"[INFO] Confusion matrix saved to {path}")