import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path adjustment
sys.path.append(os.path.join(os.path.dirname(__file__), 'deep_learning_pipeline'))

if __name__ == "__main__":
    print("=" * 50)
    print(" Running Transformer / LLM Fine-Tuning Pipeline...")
    print("=" * 50)

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Sample Input Batch
    texts = ["Sample text input for classification", "Another evaluation sentence"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    outputs = model(**inputs)
    print("Transformer Logits Output Shape:", outputs.logits.shape)
    print("\nTransformer Fine-Tuning Pipeline Initialized Successfully!\n")