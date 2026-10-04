import json
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def evaluate():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    metrics_file = params["evaluate"]["metrics_file"]

    model = tf.keras.models.load_model("models/model.h5")

    test_data = np.load("data/processed/test.npz")
    x_test, y_test = test_data["x"], test_data["y"]

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc)
    }

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics written to {metrics_file}")

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    os.makedirs("reports", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Fashion-MNIST Confusion Matrix")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=300)
    plt.close()
    print("Confusion matrix saved to reports/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()