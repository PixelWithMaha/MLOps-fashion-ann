import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]
    
    test_size = params.get("test_size", 0.1)
    seed = params.get("seed", 42)

    raw_train = np.load("data/raw/train_raw.npz")
    raw_test = np.load("data/raw/test_raw.npz")

    x_train_raw, y_train_raw = raw_train["x"], raw_train["y"]
    x_test_raw, y_test_raw = raw_test["x"], raw_test["y"]

    x_train_norm = x_train_raw / 255.0
    x_test_norm = x_test_raw / 255.0

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_norm,
        y_train_raw,
        test_size=test_size,
        random_state=seed,
        stratify=y_train_raw
    )
    
    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(
        "data/processed/train.npz",
        x=x_train,
        y=y_train
    )
    np.savez_compressed(
        "data/processed/val.npz",
        x=x_val,
        y=y_val
    )
    np.savez_compressed(
        "data/processed/test.npz",
        x=x_test_norm,
        y=y_test_raw
    )

    print(f"Data preprocessing complete:")
    print(f" - Train split: {x_train.shape[0]} samples")
    print(f" - Val split:   {x_val.shape[0]} samples (test_size={test_size}, seed={seed})")
    print(f" - Test split:  {x_test_norm.shape[0]} samples")
    print("Processed arrays saved to data/processed/")

if __name__ == "__main__":
    preprocess()