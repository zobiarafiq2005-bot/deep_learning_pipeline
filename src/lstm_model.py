import tensorflow as tf
from tensorflow.keras import layers, Model

class LSTMClassifier(Model):
    def __init__(self, vocab_size=1000, embed_dim=128, hidden_units=128, max_length=100, num_classes=1):
        super(LSTMClassifier, self).__init__()
        self.embedding = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_length)
        self.lstm_layer = layers.LSTM(units=hidden_units)
        self.dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(units=num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm_layer(x)
        x = self.dropout(x)
        return self.output_layer(x)