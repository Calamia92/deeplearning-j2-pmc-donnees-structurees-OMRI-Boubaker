import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Pipeline choice: split first, then fit the scaler on X_train only.
# Reason: fitting StandardScaler before the split would use validation/test statistics
# during preprocessing, which leaks information from data the model should not see.
# The scaler is therefore fitted on X_train, then reused to transform train/val/test.


def main() -> None:
    housing = fetch_california_housing(data_home="data")
    x, y = housing.data, housing.target

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_val_norm = scaler.transform(x_val)
    x_test_norm = scaler.transform(x_test)

    print("Pipeline California Housing: load -> split -> scale -> train -> evaluate")
    print(f"X_train shape : {x_train_norm.shape}")
    print(f"X_val shape : {x_val_norm.shape}")
    print(f"X_test shape : {x_test_norm.shape}")
    print(f"y_train shape : {y_train.shape}")
    print(f"y_val shape : {y_val.shape}")
    print(f"y_test shape : {y_test.shape}")
    print()
    print("Feature names :")
    print(housing.feature_names)
    print(f"Number of features : {len(housing.feature_names)}")
    print()
    print("X_train_norm mean (par feature) :")
    print(np.round(x_train_norm.mean(axis=0), 6))
    print("X_train_norm std (par feature) :")
    print(np.round(x_train_norm.std(axis=0), 6))


if __name__ == "__main__":
    main()
