import argparse
import datetime
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from phase7_wine_quality_baseline import load_wine


# BatchNorm order:
# Main configuration uses Dense(linear) -> BatchNormalization -> ReLU -> Dropout.
# This follows the original Batch Normalization intuition: normalize the pre-
# activation values before applying the non-linearity, so ReLU receives better
# conditioned inputs. I also compare Dense+ReLU -> BatchNormalization because the
# ordering can change val_loss and a single convention should not be trusted
# without measurement.


SEED = 42


def set_seed(seed=SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def prepare_wine_data(seed=SEED):
    df = load_wine()
    x = df.drop(["quality", "quality_3class"], axis=1).values
    y = df["quality_3class"].values
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def add_hidden_block(model, units, input_shape=None, use_batchnorm=False, bn_before_activation=True):
    dense_kwargs = {
        "units": units,
        "kernel_regularizer": regularizers.l2(0.01),
    }
    if input_shape is not None:
        dense_kwargs["input_shape"] = input_shape

    if use_batchnorm and bn_before_activation:
        model.add(layers.Dense(**dense_kwargs))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
    else:
        model.add(layers.Dense(activation="relu", **dense_kwargs))
        if use_batchnorm:
            model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))


def build_wine_model(use_batchnorm=False, bn_before_activation=True, extra_layer=False):
    """
    PMC multiclass Wine Quality.
    bn_before_activation=True: Dense lineaire -> BN -> ReLU.
    bn_before_activation=False: Dense+ReLU -> BN.
    extra_layer: ajoute une 3e couche cachee (16 units).
    """
    model = keras.Sequential()
    add_hidden_block(
        model,
        units=64,
        input_shape=(11,),
        use_batchnorm=use_batchnorm,
        bn_before_activation=bn_before_activation,
    )
    add_hidden_block(
        model,
        units=32,
        use_batchnorm=use_batchnorm,
        bn_before_activation=bn_before_activation,
    )
    if extra_layer:
        add_hidden_block(
            model,
            units=16,
            use_batchnorm=use_batchnorm,
            bn_before_activation=bn_before_activation,
        )
    model.add(layers.Dense(3, activation="softmax"))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_configs():
    return {
        "sans_bn": lambda: build_wine_model(use_batchnorm=False),
        "bn_avant_activation": lambda: build_wine_model(
            use_batchnorm=True,
            bn_before_activation=True,
        ),
        "bn_apres_activation": lambda: build_wine_model(
            use_batchnorm=True,
            bn_before_activation=False,
        ),
        "bn_extra_couche": lambda: build_wine_model(
            use_batchnorm=True,
            bn_before_activation=True,
            extra_layer=True,
        ),
    }


def train_config(name, build_fn, x_train, y_train, epochs, batch_size, seed=SEED, tensorboard=True):
    set_seed(seed)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        )
    ]
    log_dir = None
    if tensorboard:
        log_dir = f"logs/wine/{name}_{datetime.datetime.now().strftime('%H%M%S')}"
        callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1))

    model = build_fn()
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    result = {
        "val_accuracy": float(max(history.history["val_accuracy"])),
        "epochs": len(history.history["val_loss"]),
        "first_30_val_loss": float(history.history["val_loss"][min(29, len(history.history["val_loss"]) - 1)]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "log_dir": log_dir,
    }
    print(
        f"{name}: val_accuracy={result['val_accuracy']:.4f}, "
        f"stopped at epoch {result['epochs']}, "
        f"val_loss@30={result['first_30_val_loss']:.4f}, "
        f"log_dir={log_dir}"
    )
    return result


def run_four_config_comparison(x_train, y_train, epochs=200, batch_size=32, seed=SEED):
    results = {}
    for name, build_fn in make_configs().items():
        results[name] = train_config(
            name,
            build_fn,
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            tensorboard=True,
        )
    return results


def run_batch_size_one_case(x_train, y_train, epochs):
    print("\nCase limite - BatchNorm avec batch_size=1")
    return train_config(
        "bn_batch_size_1",
        make_configs()["bn_avant_activation"],
        x_train,
        y_train,
        epochs=epochs,
        batch_size=1,
        seed=SEED,
        tensorboard=False,
    )


def run_seed_adversarial(epochs=80):
    print("\nAdversarial - BN avant vs apres activation sur 5 seeds")
    wins = {"bn_avant_activation": 0, "bn_apres_activation": 0}
    for seed in range(5):
        x_train, _, y_train, _ = prepare_wine_data(seed=seed)
        before = train_config(
            f"seed_{seed}_bn_avant",
            make_configs()["bn_avant_activation"],
            x_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            seed=seed,
            tensorboard=False,
        )
        after = train_config(
            f"seed_{seed}_bn_apres",
            make_configs()["bn_apres_activation"],
            x_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            seed=seed,
            tensorboard=False,
        )
        winner = (
            "bn_avant_activation"
            if before["val_accuracy"] >= after["val_accuracy"]
            else "bn_apres_activation"
        )
        wins[winner] += 1
        print(
            f"seed={seed}: before={before['val_accuracy']:.4f}, "
            f"after={after['val_accuracy']:.4f}, winner={winner}"
        )
    print(f"Seed comparison wins : {wins}")
    print("Conclusion: if the winner changes by seed, single-seed comparisons are fragile.")
    return wins


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 8 Wine BatchNorm comparison.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--run-batch-size-one", action="store_true")
    parser.add_argument("--run-seed-adversarial", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x_wine_train, _, y_wine_train, _ = prepare_wine_data(seed=SEED)
    results = run_four_config_comparison(
        x_wine_train,
        y_wine_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=SEED,
    )
    print("\nSummary:")
    for name, result in results.items():
        print(result | {"config": name})

    if args.run_batch_size_one:
        run_batch_size_one_case(x_wine_train, y_wine_train, epochs=30)

    if args.run_seed_adversarial:
        run_seed_adversarial(epochs=80)
