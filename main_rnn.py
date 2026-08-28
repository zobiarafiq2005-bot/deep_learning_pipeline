import sys
import os
import numpy as np
import tensorflow as tf

# Deep learning pipeline path setup
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_learning_pipeline'))

from src.rnn_model import RNNClassifier

if __name__ == "__main__":
    print("=" * 50)
    print(" Running OOP-Based Simple RNN Pipeline...")
    print("=" * 50)

    # Dummy Sequence Data
    X_dummy = np.random.randint(0, 1000, size=(100, 100))
    y_dummy = np.random.randint(0, 2, size=(100, 1))

    # Instantiate OOP Model Class
    model = RNNClassifier(vocab_size=1000, embed_dim=128, hidden_units=128, max_length=100, num_classes=1)

    # Compile Model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Build model explicitly to enable summary
    model.build(input_shape=(None, 100))
    model.summary()

    # Model Training
    model.fit(X_dummy, y_dummy, epochs=3, batch_size=16)
    print("\n[SUCCESS] Simple RNN Pipeline (OOP Standard) Completed Successfully!\n")