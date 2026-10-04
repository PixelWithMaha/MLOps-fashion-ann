import os
import numpy as np
from sklearn.model_selection import train_test_split


def preprocess():
    raw_train = np.load("data/raw/train_raw.npz")
    raw_test = np.load("data/raw/test_raw.npz")

    x_train_raw, y_train_raw = raw_train["x"], raw_train["y"]
    x_test_raw, y_test_raw = raw_test["x"], raw_test["y"]

    mean = np.mean(x_train_raw)
    std = np.std(x_train_raw)
    x_train_norm = (x_train_raw - mean) / std
    x_test_norm = (x_test_raw - mean) / std

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm,
        y_train_raw,
        test_size=0.2,
        random_state=42,
        stratify=y_train_raw
    )

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_train, y=y_train)
    np.savez_compressed("data/processed/val.npz", x=x_val, y=y_val)
    np.savez_compressed("data/processed/test.npz", x=x_test_norm, y=y_test_raw)
    print("Data preprocessed using Mean/Std Standardization.")


if __name__ == "__main__":
    preprocess()