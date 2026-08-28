import os
import sys

# Ensure src modules are discoverable
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import TabularDataLoader
from ann_model import ANNClassifier
from utils import PlotUtils


def run_ann_pipeline():
    print("==================================================")
    print("      TASK 2: ANN IMPLEMENTATION PIPELINE         ")
    print("==================================================\n")

    # Step 1: Load Data
    data_loader = TabularDataLoader(n_samples=5000, n_features=12)
    X_train, X_test, y_train, y_test = data_loader.get_processed_data()

    # Step 2: Initialize & Train Model
    ann = ANNClassifier(input_dim=X_train.shape[1])
    history = ann.train(X_train, y_train, X_test, y_test, epochs=30, batch_size=32)

    # Step 3: Evaluate Model
    ann.evaluate(X_test, y_test)

    # Step 4: Save Plots & Visualizations
    PlotUtils.plot_training_history(history)
    y_preds = (ann.model.predict(X_test) > 0.5).astype(int)
    PlotUtils.plot_confusion_matrix(y_test, y_preds)

    # Step 5: Save Model Artifact
    ann.save_model("models/ann_model.keras")

    print("\n==================================================")
    print("      ANN PIPELINE COMPLETED SUCCESSFULLY!        ")
    print("==================================================")


if __name__ == "__main__":
    run_ann_pipeline()