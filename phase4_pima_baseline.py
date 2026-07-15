import argparse
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# Class distribution before training:
# np.bincount(y) = [500, 268], so class 0 is 65.1% and class 1 is 34.9%.
# A naive model that always predicts the majority class 0 would already reach
# 500 / 768 = 65.1% accuracy. The baseline must beat that and predict positives.


SEED = 42
PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
COLS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]
SUSPECT_ZERO_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_pima() -> pd.DataFrame:
    return pd.read_csv(PIMA_URL, names=COLS)


def print_data_diagnostics(df: pd.DataFrame) -> None:
    class_counts = df["Outcome"].value_counts().sort_index()
    class_percentages = class_counts / len(df) * 100

    print("Distribution classes :")
    for label, count in class_counts.items():
        print(f"{label} {count} ({class_percentages[label]:.1f}%)")

    majority_accuracy = class_counts.max() / len(df)
    print(f"Majority-class baseline accuracy : {majority_accuracy:.4f}")
    print()
    print("Colonnes avec des zeros suspects :")
    print((df == 0).sum())
    print()


def impute_suspect_zeros(df: pd.DataFrame) -> pd.DataFrame:
    df_imputed = df.copy()
    for col in SUSPECT_ZERO_COLS:
        non_zero_median = df_imputed.loc[df_imputed[col] != 0, col].median()
        df_imputed[col] = df_imputed[col].replace(0, non_zero_median)
    return df_imputed


def prepare_data(df: pd.DataFrame):
    x = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

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


def build_binary_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_baseline(df: pd.DataFrame, epochs: int, class_weight=None, verbose: int = 0):
    set_seed()
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_data(df)
    model = build_binary_model(input_dim=x_train.shape[1])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        verbose=verbose,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    val_predictions_mean = float(model.predict(x_val, verbose=0).mean())
    test_predictions_mean = float(model.predict(x_test, verbose=0).mean())
    best_val_accuracy = float(max(history.history["val_accuracy"]))

    return {
        "history": history,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "val_predictions_mean": val_predictions_mean,
        "test_predictions_mean": test_predictions_mean,
    }


def print_results(name: str, results: dict) -> None:
    print(f"{name} best val_accuracy : {results['best_val_accuracy']:.4f}")
    print(f"{name} final val_accuracy : {results['final_val_accuracy']:.4f}")
    print(f"{name} test_accuracy : {results['test_accuracy']:.4f}")
    print(f"{name} model.predict(X_val).mean() : {results['val_predictions_mean']:.4f}")
    print(f"{name} model.predict(X_test).mean() : {results['test_predictions_mean']:.4f}")
    print()


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 4 Pima binary classification baseline.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--verbose", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_pima()
    print_data_diagnostics(df)

    baseline_results = train_baseline(df, epochs=args.epochs, verbose=args.verbose)
    print_results("Baseline", baseline_results)

    df_imputed = impute_suspect_zeros(df)
    imputed_results = train_baseline(df_imputed, epochs=args.epochs, verbose=args.verbose)
    print_results("Median-imputed", imputed_results)
    delta = imputed_results["best_val_accuracy"] - baseline_results["best_val_accuracy"]
    print(f"Median imputation val_accuracy delta : {delta:+.4f}")

    if baseline_results["test_predictions_mean"] < 0.10:
        weighted_results = train_baseline(
            df,
            epochs=args.epochs,
            class_weight={0: 1.0, 1: 1.9},
            verbose=args.verbose,
        )
        print("Baseline predicted too few positives; class_weight retry:")
        print_results("Class-weighted", weighted_results)
    else:
        print("Prediction mean check passed: the baseline is not only predicting class 0.")
