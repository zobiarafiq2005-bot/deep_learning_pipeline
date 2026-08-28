import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TabularDataLoader:
    """Class to load, preprocess, and scale tabular data for ANN training."""

    def __init__(self, n_samples: int = 5000, n_features: int = 12, random_state: int = 42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.scaler = StandardScaler()

    def generate_or_load_data(self):
        """Generates a synthetic binary classification dataset for ANN training."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=8,
            n_redundant=4,
            random_state=self.random_state
        )
        return X, y

    def get_processed_data(self, test_size: float = 0.2):
        """Splits and scales features for neural network ingestion."""
        X, y = self.generate_or_load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"[INFO] Data Loaded. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test