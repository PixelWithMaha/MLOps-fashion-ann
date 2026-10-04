import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf

def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    train_data = np.load("data/processed/train.npz")
    x_train, y_train = train_data["x"], train_data["y"]

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(params["hidden_units_1"], activation="relu"),
        tf.keras.layers.Dropout(params.get("dropout_rate", 0.2)),
        tf.keras.layers.Dense(params["hidden_units_2"], activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train,
        y_train,
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        validation_split=0.1,
        verbose=1
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/model.h5")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv("models/history.csv", index=False)
    print("Model trained and saved to models/model.h5. History saved to models/history.csv.")

if __name__ == "__main__":
    train()