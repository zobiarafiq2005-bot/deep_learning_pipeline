import tensorflow as tf
from tensorflow.keras import layers, models

class CNNModelBuilder:
    """
    Modular CNN Architecture for Image Classification tasks.
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        """
        Builds a standard Convolutional Neural Network with Conv2D,
        MaxPooling2D, Dropout, and Dense layers.
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Classification Head
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax' if self.num_classes > 2 else 'sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy' if self.num_classes > 2 else 'binary_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("[INFO] CNN Model Built Successfully!")
        return self.model