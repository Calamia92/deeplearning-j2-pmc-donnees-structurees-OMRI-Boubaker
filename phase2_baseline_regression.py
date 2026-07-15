import argparse
import random

import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


SEED = 42


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def load_california_data():
    housing = fetch_california_housing(data_home="data")
    x, y = housing.data, housing.target

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        random_state=SEED,
    )

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_val_norm = scaler.transform(x_val)
    x_test_norm = scaler.transform(x_test)

    return (x_train, x_val, x_test), (x_train_norm, x_val_norm, x_test_norm), (
        y_train,
        y_val,
        y_test,
    )


def build_regression_model(input_dim, optimizer="adam"):
    # Do not use sigmoid on the output layer for this regression task.
    # A sigmoid would force predictions into [0, 1], while California Housing
    # targets are continuous median prices that commonly exceed 1.0
    # (hundreds of thousands of dollars). The model could train but would be
    # unable to represent valid target values.
    model = keras.Sequential(
        [
            layers.Dense(64, activation="relu", input_shape=(input_dim,)),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_and_evaluate(
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    epochs,
    batch_size,
    optimizer="adam",
    verbose=1,
    show_summary=True,
):
    set_seed()
    model = build_regression_model(input_dim=x_train.shape[1], optimizer=optimizer)
    if show_summary:
        model.summary()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        verbose=verbose,
    )
    test_loss, test_mae = model.evaluate(x_test, y_test, verbose=0)
    return model, history, test_loss, test_mae


def first_epoch_under(history, threshold):
    for index, value in enumerate(history.history["val_loss"], start=1):
        if value < threshold:
            return index
    return None


def run_batch_size_checks(x_train, x_val, x_test, y_train, y_val, y_test, epochs):
    print("\nEdge case - batch size comparison")
    for label, batch_size in [
        ("Stochastic updates, batch_size=1", 1),
        ("Batch GD, batch_size=len(X_train_norm)", len(x_train)),
    ]:
        _, history, _, test_mae = train_and_evaluate(
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            show_summary=False,
        )
        print(
            f"{label} : first val_loss={history.history['val_loss'][0]:.4f}, "
            f"last val_loss={history.history['val_loss'][-1]:.4f}, "
            f"test_mae={test_mae:.4f}"
        )
    print(
        "Observation : batch_size=1 is noisier but converges in fewer epochs here; "
        "full batch is smoother but much slower per parameter update."
    )


def run_adversarial_checks(raw_data, norm_data, targets, epochs):
    x_train_raw, x_val_raw, x_test_raw = raw_data
    x_train_norm, x_val_norm, x_test_norm = norm_data
    y_train, y_val, y_test = targets

    print("\nAdversarial check - training without normalization")
    _, raw_history, _, raw_test_mae = train_and_evaluate(
        x_train_raw,
        y_train,
        x_val_raw,
        y_val,
        x_test_raw,
        y_test,
        epochs=epochs,
        batch_size=32,
        verbose=0,
        show_summary=False,
    )
    print(
        f"Raw features MSE starts at {raw_history.history['loss'][0]:.4f} "
        f"and ends at {raw_history.history['loss'][-1]:.4f}; "
        f"test_mae={raw_test_mae:.4f}"
    )

    print("\nAdversarial check - Adam vs SGD, lr=0.001 on normalized data")
    optimizers = [
        ("Adam", keras.optimizers.Adam(learning_rate=0.001)),
        ("SGD", keras.optimizers.SGD(learning_rate=0.001)),
    ]
    for label, optimizer in optimizers:
        _, history, _, test_mae = train_and_evaluate(
            x_train_norm,
            y_train,
            x_val_norm,
            y_val,
            x_test_norm,
            y_test,
            epochs=epochs,
            batch_size=32,
            optimizer=optimizer,
            verbose=0,
            show_summary=False,
        )
        epoch_under_05 = first_epoch_under(history, threshold=0.5)
        print(
            f"{label} : first val_loss<0.5 at epoch {epoch_under_05}, "
            f"last val_loss={history.history['val_loss'][-1]:.4f}, "
            f"test_mae={test_mae:.4f}"
        )
    print("Observation : Adam usually converges in fewer epochs than SGD at the same lr.")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2 California Housing regression baseline.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-edge-checks", action="store_true")
    parser.add_argument("--run-adversarial-checks", action="store_true")
    parser.add_argument("--checks-epochs", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_data, norm_data, targets = load_california_data()
    x_train_norm, x_val_norm, x_test_norm = norm_data
    y_train, y_val, y_test = targets

    _, history, test_loss, test_mae = train_and_evaluate(
        x_train_norm,
        y_train,
        x_val_norm,
        y_val,
        x_test_norm,
        y_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

    print(f"MAE test : {test_mae:.4f} (en centaines de milliers de $)")
    print(f"MSE test : {test_loss:.4f}")
    epoch_index = min(49, len(history.history["val_loss"]) - 1)
    print(
        f"Baseline val_loss : first={history.history['val_loss'][0]:.4f}, "
        f"epoch_{epoch_index + 1}={history.history['val_loss'][epoch_index]:.4f}, "
        f"last={history.history['val_loss'][-1]:.4f}"
    )

    if args.run_edge_checks:
        run_batch_size_checks(
            x_train_norm,
            x_val_norm,
            x_test_norm,
            y_train,
            y_val,
            y_test,
            epochs=args.checks_epochs,
        )

    if args.run_adversarial_checks:
        run_adversarial_checks(raw_data, norm_data, targets, epochs=args.checks_epochs)
