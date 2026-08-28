import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

class VGG16TransferLearningBuilder:
    """
    Transfer Learning Model Builder using VGG16 pre-trained on ImageNet.
    """
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, freeze_base=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.freeze_base = freeze_base
        self.model = None

    def build_model(self):
        """
        Loads pre-trained VGG16 base model, attaches custom classification head.
        """
        # Load pre-trained VGG16 without the top dense layers
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )

        # Freeze pre-trained convolutional base layers if specified
        if self.freeze_base:
            base_model.trainable = False

        # Build fine-tuning / classification head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = models.Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        print("[INFO] VGG16 Transfer Learning Model Built Successfully!")
        return self.model