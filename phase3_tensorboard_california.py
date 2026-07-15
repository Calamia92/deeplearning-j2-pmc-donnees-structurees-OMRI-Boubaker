import argparse
import datetime

from tensorflow import keras

from phase2_baseline_regression import build_regression_model, load_california_data, set_seed


def train_with_tensorboard(x_train, y_train, x_val, y_val, run_name, epochs=100):
    """Entraine un modele de regression avec un callback TensorBoard horodate."""
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    log_dir = f"logs/fit/{run_name}_{timestamp}"
    tb_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    set_seed()
    model = build_regression_model(input_dim=8)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[tb_callback],
        verbose=0,
    )

    print(f"Run '{run_name}' termine. Logs dans {log_dir}")
    print(
        f"{run_name} loss : train {history.history['loss'][0]:.4f} -> "
        f"{history.history['loss'][-1]:.4f}, val {history.history['val_loss'][0]:.4f} -> "
        f"{history.history['val_loss'][-1]:.4f}"
    )
    return model, history


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 3 TensorBoard California Housing.")
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raw_data, norm_data, targets = load_california_data()
    x_train, x_val, _ = raw_data
    x_train_norm, x_val_norm, _ = norm_data
    y_train, y_val, _ = targets

    model_norm, history_norm = train_with_tensorboard(
        x_train_norm,
        y_train,
        x_val_norm,
        y_val,
        run_name="california_norm",
        epochs=args.epochs,
    )

    model_raw, history_raw = train_with_tensorboard(
        x_train,
        y_train,
        x_val,
        y_val,
        run_name="california_raw",
        epochs=args.epochs,
    )

    print("Lancer TensorBoard : tensorboard --logdir=logs/fit")
    print("Puis ouvrir http://localhost:6006")


# Diagnostic Phase 3:
# Situation (a) pour california_norm: train_loss et val_loss descendent ensemble,
# ce qui indique un pipeline sain.
# california_raw a une loss initiale beaucoup plus haute et des courbes plus
# instables: meme architecture, meme optimizer, mais les gradients sont moins
# bien conditionnes sans normalisation.
# Si TensorBoard affiche "No dashboards are active", verifier d'abord le chemin
# logs/fit. Si le port 6006 est occupe par une autre instance TensorBoard,
# arreter l'ancienne instance avant de relancer.
