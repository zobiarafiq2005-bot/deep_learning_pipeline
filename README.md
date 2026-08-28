```text
deep_learning_pipeline/
│
├── src/                          # Modular Core Package
│   ├── __init__.py               # Package initializer
│   ├── data_loader.py            # Data ingestion & scaling utilities
│   ├── ann_model.py              # OOP Modular ANN Architecture
│   ├── cnn_model.py              # Custom Multi-layer CNN Architecture
│   ├── vgg16_model.py            # VGG16 Transfer Learning Builder
│   ├── rnn_model.py              # Simple RNN Architecture
│   ├── lstm_model.py             # LSTM Architecture
│   ├── predictor.py              # Production Inference Engine
│   └── utils.py                  # Evaluation metrics & visualization utilities
│
├── models/                       # Model Artifact Exports (.keras)
│   ├── ann_model.keras
│   ├── cnn_model.keras
│   ├── vgg16_model.keras
│   └── vgg16_finetuned_model.keras
│
├── outputs/figures/              # Generated Visualizations & Evaluation Plots
│   ├── ann_training_curves.png
│   ├── ann_confusion_matrix.png
│   ├── cnn_training_curves.png
│   ├── vgg16_training_curves.png
│   └── vgg16_finetuned_curves.png
│
├── main_ann.py                   # Day 1 Execution Script (ANN)
├── main_cnn.py                   # Day 2 Execution Script (CNN)
├── main_vgg16.py                 # Day 3 Execution Script (Transfer Learning)
├── main_finetune.py              # Day 4 Execution Script (Fine-Tuning & Metrics)
├── main_inference.py             # Day 5 Execution Script (Production Predictor)
├── main_rnn.py                   # RNN Pipeline Execution Script
├── main_lstm.py                  # LSTM Pipeline Execution Script
├── main_transformer.py           # Transformer / LLM Fine-Tuning Script
│
├── .gitignore                    # Git exclusion rules
├── requirements.txt              # Python dependencies
├── README.md                     # Main project documentation
└── TASK2_DOCUMENTATION.md        # Task 2 Comprehensive Report
```