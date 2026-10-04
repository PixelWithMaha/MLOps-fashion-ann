import os
import numpy as np
import tensorflow as tf

def prepare():
    os.makedirs("data/raw", exist_ok=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    np.savez_compressed("data/raw/train_raw.npz", x=x_train, y=y_train)
    np.savez_compressed("data/raw/test_raw.npz", x=x_test, y=y_test)
    print("Raw data downloaded and saved to data/raw/")

if __name__ == "__main__":
    prepare()