import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from phase4_pima_baseline import load_pima, prepare_data


# Target for Phase 5:
# Improve class-1 recall by at least 5 percentage points over the Phase 4
# unregularized baseline, or at minimum keep validation accuracy competitive
# while reducing overfitting.
#
# Levers, in order:
# 1. L2: first choice because it directly limits weight magnitude without adding
#    stochastic noise, useful on a small tabular dataset.
# 2. Early stopping: stops training when validation loss no longer improves and
#    restores the best epoch.
# 3. Dropout: tested after L2 because it is stronger/noisier and can be
#    redundant on a dataset with only 768 rows.
# 4. class_weight / decision threshold: reserved for the next iteration if class
#    1 recall remains too low despite regularization.


SEED = 42


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def build_pima_regularized(l2_lambda=0.01, use_dropout=False):
    """
    Modele Pima avec regularisation L2 optionnelle et Dropout optionnel.
    Si use_dropout=True, insere un Dropout(0.3) apres chaque couche cachee.
    """
    model = keras.Sequential()
    model.add(
        layers.Dense(
            64,
            activation="relu",
            input_shape=(8,),
            kernel_regularizer=regularizers.l2(l2_lambda),
        )
    )
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(
        layers.Dense(
            32,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_lambda),
        )
    )
    if use_dropout:
        model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def make_early_stopping(patience):
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def train_config(name, x_train, y_train, x_val, y_val, l2_lambda, use_dropout, patience, epochs):
    set_seed()
    model = build_pima_regularized(l2_lambda=l2_lambda, use_dropout=use_dropout)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_val, y_val),
        batch_size=32,
        callbacks=[make_early_stopping(patience)],
        verbose=0,
    )

    probabilities = model.predict(x_val, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(int)
    result = {
        "name": name,
        "model": model,
        "history": history,
        "epochs": len(history.history["val_loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "val_accuracy": float(accuracy_score(y_val, predictions)),
        "class1_recall": float(recall_score(y_val, predictions)),
        "macro_f1": float(f1_score(y_val, predictions, average="macro")),
        "prediction_mean": float(probabilities.mean()),
    }
    print(
        f"{name} : stop_epoch={result['epochs']}, "
        f"best_val_accuracy={result['best_val_accuracy']:.4f}, "
        f"class1_recall={result['class1_recall']:.4f}, "
        f"macro_f1={result['macro_f1']:.4f}, "
        f"prediction_mean={result['prediction_mean']:.4f}"
    )
    return result


def plot_val_losses(results, output_path="phase5_pima_3configs.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for axis, result in zip(axes, results):
        val_loss = result["history"].history["val_loss"]
        axis.plot(val_loss, label="val_loss")
        axis.axvline(result["epochs"] - 1, color="red", linestyle="--", label="stop")
        axis.set_title(result["name"])
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation loss")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Courbes sauvegardees dans {output_path}")


def run_patience_edge_case(x_train, y_train, x_val, y_val):
    print("\nEdge case - patience=1")
    return train_config(
        "L2 patience=1",
        x_train,
        y_train,
        x_val,
        y_val,
        l2_lambda=0.01,
        use_dropout=False,
        patience=1,
        epochs=300,
    )


def run_l2_adversarial(x_train, y_train, x_val, y_val):
    print("\nAdversarial - l2_lambda=10.0")
    result = train_config(
        "L2=10.0",
        x_train,
        y_train,
        x_val,
        y_val,
        l2_lambda=10.0,
        use_dropout=False,
        patience=15,
        epochs=300,
    )
    first_layer_weights = result["model"].layers[0].get_weights()[0]
    print(f"L2=10 first layer weight mean : {first_layer_weights.mean():.8f}")
    print(f"L2=10 first layer weight abs mean : {np.abs(first_layer_weights).mean():.8f}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 5 Pima regularization experiments.")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--run-edge-checks", action="store_true")
    parser.add_argument("--run-adversarial-checks", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_pima()
    x_train, x_val, _, y_train, y_val, _ = prepare_data(df)

    results = [
        train_config(
            "Baseline",
            x_train,
            y_train,
            x_val,
            y_val,
            l2_lambda=0.0,
            use_dropout=False,
            patience=args.patience,
            epochs=args.epochs,
        ),
        train_config(
            "L2 seul",
            x_train,
            y_train,
            x_val,
            y_val,
            l2_lambda=0.01,
            use_dropout=False,
            patience=args.patience,
            epochs=args.epochs,
        ),
        train_config(
            "L2 + Dropout",
            x_train,
            y_train,
            x_val,
            y_val,
            l2_lambda=0.01,
            use_dropout=True,
            patience=args.patience,
            epochs=args.epochs,
        ),
    ]
    plot_val_losses(results)

    baseline_recall = results[0]["class1_recall"]
    best_recall = max(result["class1_recall"] for result in results[1:])
    print(f"Recall class 1 delta vs baseline : {best_recall - baseline_recall:+.4f}")
    best_macro_f1 = max(result["macro_f1"] for result in results)
    print(f"Best macro F1 : {best_macro_f1:.4f}")

    if args.run_edge_checks:
        run_patience_edge_case(x_train, y_train, x_val, y_val)

    if args.run_adversarial_checks:
        run_l2_adversarial(x_train, y_train, x_val, y_val)
