import argparse
import random

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from phase4_pima_baseline import load_pima, prepare_data


# Tuner guardrail:
# I use 100 epochs with EarlyStopping(patience=10) for the real 15-trial search.
# This is long enough for slower configurations to reveal whether they keep
# improving, while early stopping prevents wasting epochs on clear plateaus.
# I compare it with a short run through --run-short-comparison: if the best
# architecture or best val_accuracy changes materially, max_epochs was too low.


SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_pima_model(hp):
    """
    HyperModel Pima : keras-tuner echantillonne les valeurs dans l'espace defini ici.
    hp.Int, hp.Float, hp.Choice sont les trois types d'hyperparametres supportes.
    Chaque appel construit un modele avec une configuration differente.
    """
    model = keras.Sequential()
    units_1 = hp.Int("units_1", min_value=32, max_value=128, step=32)
    units_2 = hp.Int("units_2", min_value=16, max_value=64, step=16)
    activation = hp.Choice("activation", values=["relu", "tanh"])
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

    model.add(layers.Dense(units_1, activation=activation, input_shape=(8,)))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units_2, activation=activation))
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pima_bad_lr_model(hp):
    model = keras.Sequential()
    units_1 = hp.Int("units_1", min_value=32, max_value=128, step=32)
    units_2 = hp.Int("units_2", min_value=16, max_value=64, step=16)
    learning_rate = hp.Float("learning_rate", min_value=10.0, max_value=100.0)
    model.add(layers.Dense(units_1, activation="relu", input_shape=(8,)))
    model.add(layers.Dense(units_2, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_tuner(build_fn, seed, max_trials, project_name, overwrite=True):
    return kt.RandomSearch(
        build_fn,
        objective="val_accuracy",
        max_trials=max_trials,
        seed=seed,
        directory="tuning_pima",
        project_name=project_name,
        overwrite=overwrite,
    )


def trial_scores(tuner):
    trials = tuner.oracle.trials.values()
    scores = [trial.score for trial in trials if trial.score is not None]
    return scores


def run_search(
    x_train,
    y_train,
    seed=SEED,
    max_trials=15,
    epochs=100,
    project_name="pima_random",
    build_fn=build_pima_model,
):
    set_seed(seed)
    tuner = make_tuner(
        build_fn=build_fn,
        seed=seed,
        max_trials=max_trials,
        project_name=project_name,
    )
    tuner.search_space_summary()
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10)
    tuner.search(
        x_train,
        y_train,
        epochs=epochs,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )

    best_hp = tuner.get_best_hyperparameters()[0]
    print("Meilleur learning_rate :", best_hp.get("learning_rate"))
    print("Meilleures units_1 :", best_hp.get("units_1"))
    print("Meilleures units_2 :", best_hp.get("units_2"))
    if "activation" in best_hp.values:
        print("Meilleure activation :", best_hp.get("activation"))
    if "dropout_rate" in best_hp.values:
        print("Meilleur dropout_rate :", best_hp.get("dropout_rate"))

    tuner.results_summary(num_trials=5)
    print("Top 5 hyperparameters:")
    for index, hp in enumerate(tuner.get_best_hyperparameters(num_trials=5), start=1):
        print(f"{index}. {hp.values}")

    best_model = tuner.hypermodel.build(best_hp)
    history_best = best_model.fit(
        x_train,
        y_train,
        epochs=200,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )
    best_val_accuracy = float(max(history_best.history["val_accuracy"]))
    print(f"Best model val_accuracy : {best_val_accuracy:.4f}")
    return tuner, best_hp, best_val_accuracy


def plot_seed_distributions(scores_42, scores_43, output_path="phase6_pima_seed_stability.png"):
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.boxplot([scores_42, scores_43], tick_labels=["seed=42", "seed=43"])
    axis.scatter(np.ones(len(scores_42)), scores_42, alpha=0.7)
    axis.scatter(np.ones(len(scores_43)) * 2, scores_43, alpha=0.7)
    axis.set_ylabel("Trial val_accuracy")
    axis.set_title("Pima tuner stability across seeds")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Distribution plot saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 6 Pima keras-tuner RandomSearch.")
    parser.add_argument("--max-trials", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-edge-check", action="store_true")
    parser.add_argument("--run-short-comparison", action="store_true")
    parser.add_argument("--run-adversarial-check", action="store_true")
    parser.add_argument("--run-stability-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = load_pima()
    x_train, _, _, y_train, _, _ = prepare_data(df)

    tuner, best_hp, best_val_accuracy = run_search(
        x_train,
        y_train,
        seed=args.seed,
        max_trials=args.max_trials,
        epochs=args.epochs,
        project_name=f"pima_random_seed_{args.seed}_trials_{args.max_trials}_epochs_{args.epochs}",
    )

    if args.run_edge_check:
        print("\nEdge case - max_trials=1")
        _, _, edge_accuracy = run_search(
            x_train,
            y_train,
            seed=args.seed,
            max_trials=1,
            epochs=args.epochs,
            project_name=f"pima_random_edge_seed_{args.seed}",
        )
        print(f"Gain from {args.max_trials} trials vs 1 trial : {best_val_accuracy - edge_accuracy:+.4f}")

    if args.run_short_comparison:
        print("\nShort max_epochs comparison")
        _, short_hp, short_accuracy = run_search(
            x_train,
            y_train,
            seed=args.seed,
            max_trials=args.max_trials,
            epochs=10,
            project_name=f"pima_random_short_seed_{args.seed}",
        )
        print(f"Short best hp : {short_hp.values}")
        print(f"Long best hp : {best_hp.values}")
        print(f"Long vs short val_accuracy delta : {best_val_accuracy - short_accuracy:+.4f}")

    if args.run_adversarial_check:
        print("\nAdversarial - catastrophic learning-rate search space")
        bad_tuner, _, bad_accuracy = run_search(
            x_train,
            y_train,
            seed=args.seed,
            max_trials=5,
            epochs=30,
            project_name=f"pima_bad_lr_seed_{args.seed}",
            build_fn=build_pima_bad_lr_model,
        )
        print(f"Bad LR best val_accuracy : {bad_accuracy:.4f}")
        print(f"Bad LR trial scores : {trial_scores(bad_tuner)}")

    if args.run_stability_check:
        print("\nStability - seed=43")
        tuner_43, hp_43, accuracy_43 = run_search(
            x_train,
            y_train,
            seed=43,
            max_trials=args.max_trials,
            epochs=args.epochs,
            project_name=f"pima_random_seed_43_trials_{args.max_trials}_epochs_{args.epochs}",
        )
        print(f"Seed 42 best hp : {best_hp.values}")
        print(f"Seed 43 best hp : {hp_43.values}")
        print(f"Seed best val_accuracy delta : {abs(best_val_accuracy - accuracy_43):.4f}")
        plot_seed_distributions(trial_scores(tuner), trial_scores(tuner_43))
