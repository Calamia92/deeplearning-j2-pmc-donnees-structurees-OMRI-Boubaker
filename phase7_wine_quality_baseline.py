import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# Modeling choice:
# I use multiclass classification on three aggregated classes: low, medium, high.
# This gives a stable baseline despite the severe imbalance in raw quality scores.
# Alternative 1, raw 6-class classification, keeps more detail but each minority
# class has too few examples and accuracy is dominated by quality 5.
# Alternative 2, regression, respects the numeric nature of quality but treats a
# wrong prediction of 4 vs 5 the same kind of output problem as 4 vs 8 during
# thresholding decisions.
# Alternative 3, ordinal classification, would preserve the order 3 < ... < 8
# more rigorously, but it is outside the basic Keras softmax pipeline here.


SEED = 42
WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def map_quality(q: int) -> int:
    if q <= 4:
        return 0
    if q <= 6:
        return 1
    return 2


def load_wine() -> pd.DataFrame:
    df_wine = pd.read_csv(WINE_URL, sep=";")
    print("Distribution des qualites brutes :")
    print(df_wine["quality"].value_counts().sort_index())

    df_wine["quality_3class"] = df_wine["quality"].apply(map_quality)
    print("\nDistribution agregee (3 classes) :")
    print(df_wine["quality_3class"].value_counts().sort_index())
    print()
    return df_wine


def split_and_scale(x, y):
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train_val,
    )

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_val_norm = scaler.transform(x_val)
    x_test_norm = scaler.transform(x_test)
    return x_train_norm, x_val_norm, x_test_norm, y_train, y_val, y_test


def build_multiclass_model(input_dim: int, num_classes: int, loss="sparse_categorical_crossentropy"):
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss=loss, metrics=["accuracy"])
    return model


def train_multiclass(x, y, num_classes: int, epochs: int, loss="sparse_categorical_crossentropy"):
    set_seed()
    x_train, x_val, x_test, y_train, y_val, y_test = split_and_scale(x, y)
    model = build_multiclass_model(
        input_dim=x_train.shape[1],
        num_classes=num_classes,
        loss=loss,
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(x_val, y_val),
        verbose=0,
    )
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    return {
        "history": history,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "confusion_matrix": matrix,
        "predicted_counts": np.bincount(y_pred, minlength=num_classes),
    }


def print_results(name: str, result: dict) -> None:
    print(f"{name} best val_accuracy : {result['best_val_accuracy']:.4f}")
    print(f"{name} final val_accuracy : {result['final_val_accuracy']:.4f}")
    print(f"{name} test_accuracy : {result['test_accuracy']:.4f}")
    print(f"{name} predicted class counts : {result['predicted_counts']}")
    print(f"{name} confusion matrix :")
    print(result["confusion_matrix"])
    print()


def run_raw_quality_case(df_wine: pd.DataFrame, epochs: int) -> dict:
    x = df_wine.drop(["quality", "quality_3class"], axis=1).values
    raw_quality = df_wine["quality"].values
    y_raw = raw_quality - raw_quality.min()
    result = train_multiclass(x, y_raw, num_classes=len(np.unique(y_raw)), epochs=epochs)
    print_results("Raw 6-class quality", result)
    return result


def run_categorical_crossentropy_adversarial(x, y) -> None:
    print("Adversarial categorical_crossentropy check :")
    try:
        train_multiclass(
            x,
            y,
            num_classes=3,
            epochs=1,
            loss="categorical_crossentropy",
        )
    except Exception as exc:
        print(type(exc).__name__)
        print(str(exc).splitlines()[0])
        print(
            "This fails because categorical_crossentropy expects one-hot labels, "
            "while sparse_categorical_crossentropy accepts integer labels."
        )
        print()


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 7 Wine Quality multiclass baseline.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--run-raw-quality-case", action="store_true")
    parser.add_argument("--run-adversarial-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_wine()
    x_wine = df.drop(["quality", "quality_3class"], axis=1).values
    y_wine = df["quality_3class"].values

    majority_accuracy = df["quality_3class"].value_counts(normalize=True).max()
    print(f"Majority-class baseline accuracy : {majority_accuracy:.4f}")
    print()

    result_3class = train_multiclass(x_wine, y_wine, num_classes=3, epochs=args.epochs)
    print_results("Wine 3-class baseline", result_3class)

    if args.run_raw_quality_case:
        run_raw_quality_case(df, epochs=args.epochs)

    if args.run_adversarial_check:
        run_categorical_crossentropy_adversarial(x_wine, y_wine)
