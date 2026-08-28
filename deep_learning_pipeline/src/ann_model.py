import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class ANNClassifier:
    """Modular Class to build, train, evaluate, and save Artificial Neural Networks."""

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.model = self._build_architecture()

    def _build_architecture(self) -> Sequential:
        """Constructs a Multi-Layer Perceptron (ANN)."""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.input_dim,)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Binary Classification
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs: int = 50, batch_size: int = 32):
        """Trains the ANN model with EarlyStopping."""
        print("[INFO] Starting ANN Model Training...")
        early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1
        )
        return history

    def evaluate(self, X_test, y_test):
        """Evaluates model performance on unseen test set."""
        loss, accuracy, auc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\n[INFO] Test Evaluation Results:")
        print(f"       - Loss: {loss:.4f}")
        print(f"       - Accuracy: {accuracy * 100:.2f}%")
        print(f"       - AUC Score: {auc:.4f}")
        return {'loss': loss, 'accuracy': accuracy, 'auc': auc}

    def save_model(self, filepath: str = "models/ann_model.keras"):
        """Saves model architecture and trained weights."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"[INFO] ANN Model saved successfully to: {filepath}")