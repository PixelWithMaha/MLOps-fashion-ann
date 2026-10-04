#this is where preprocessing shall be done 
import os
import yaml
import numpy as np

def process():
    raw_train = np.load("data/raw/train_raw.npz")
    raw_test = np.load("data/raw/test_raw.npz")
    
    x_train = raw_train["x"] / 255.0
    y_train = raw_train["y"]
    x_test = raw_test["x"] / 255.0
    y_test = raw_test["y"]
    
    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed("data/processed/train.npz", x=x_train, y=y_train)
    np.savez_compressed("data/processed/test.npz", x=x_test, y=y_test)
    print("Data processed and saved to data/processed/")

if __name__ == "__main__":
    process()