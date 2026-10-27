"""Evaluate the saved model on the held-out test split created by train.py."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from train import make_ds

MODELS = Path("models")


def main():
    model = tf.keras.models.load_model(MODELS / "model.keras")
    classes = json.loads((MODELS / "class_names.json").read_text())
    split = json.loads((MODELS / "splits.json").read_text())
    y_true = np.array(split["test_labels"])

    probs = model.predict(make_ds(split["test_paths"], split["test_labels"], False))
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    print(f"Test images: {len(y_true)}\nAccuracy: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

    (MODELS / "metrics.json").write_text(json.dumps({"accuracy": acc, "report": report}, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(MODELS / "confusion_matrix.png", dpi=150)
    print("Saved models/metrics.json and models/confusion_matrix.png")


if __name__ == "__main__":
    main()
