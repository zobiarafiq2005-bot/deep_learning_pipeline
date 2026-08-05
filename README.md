#  Deep Learning Pipeline: End-to-End Modular Architecture

A comprehensive, production-ready Deep Learning Pipeline engineered in Python using **TensorFlow/Keras**, **Scikit-Learn**, and **Matplotlib**. This repository demonstrates modular Object-Oriented Programming (OOP) design patterns applied across standard Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), Transfer Learning, Fine-Tuning, and real-time Inference Engines.

---

##  Project Overview

This repository documents a 5-day structured development lifecycle for deep learning models on tabular and image benchmark datasets (including CIFAR-10). The project is structured with modular maintainability in mind, separating data loading, model architecture builders, visualization utilities, and inference scripts.

---

##  Repository Architecture & Directory Structure

```text
deep_learning_pipeline/
│
├── src/                         # Modular Core Package
│   ├── __init__.py              # Package initializer
│   ├── data_loader.py           # Data ingestion & scaling utilities
│   ├── ann_model.py             # OOP Modular ANN Architecture
│   ├── cnn_model.py             # Custom Multi-layer CNN Architecture
│   ├── vgg16_model.py           # VGG16 Transfer Learning Builder
│   ├── predictor.py             # Production Inference Engine
│   └── utils.py                 # Evaluation metrics & visualization utilities
│
├── models/                      # Model Artifact Exports (.keras)
│   ├── ann_model.keras
│   ├── cnn_model.keras
│   ├── vgg16_model.keras
│   └── vgg16_finetuned_model.keras
│
├── outputs/figures/             # Generated Visualizations & Evaluation Plots
│   ├── ann_training_curves.png
│   ├── ann_confusion_matrix.png
│   ├── cnn_training_curves.png
│   ├── vgg16_training_curves.png
│   └── vgg16_finetuned_curves.png
│
├── main_ann.py                  # Day 1 Execution Script (ANN)
├── main_cnn.py                  # Day 2 Execution Script (CNN)
├── main_vgg16.py                # Day 3 Execution Script (Transfer Learning)
├── main_finetune.py             # Day 4 Execution Script (Fine-Tuning & Metrics)
├── main_inference.py            # Day 5 Execution Script (Production Predictor)
│
├── .gitignore                   # Git exclusion rules
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation