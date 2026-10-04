import os
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess():
    raw_train = np.load("data/raw/train_raw.npz")
    raw_test = np.load("data/raw/test_raw.npz")

    x_train_raw, y_train_raw = raw_train["x"], raw_train["y"]
    x_test_raw, y_test_raw = raw_test["x"], raw_test["y"]

    x_train_norm = x_train_raw / 255.0
    x_test_norm = x_test_raw / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm,
        y_train_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_train_raw,
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_train, y=y_train)
    np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
    np.savez_compressed("data/processed/test.npz", x=x_test_norm, y=y_test_raw)

    print(
        f"Preprocessing complete (Min-Max Scaling):\n"
        f" - Train: {x_train.shape[0]} samples\n"
        f" - Val: {x_val.shape[0]} samples\n"
        f" - Test: {x_test_norm.shape[0]} samples"
    )


if __name__ == "__main__":
    preprocess()