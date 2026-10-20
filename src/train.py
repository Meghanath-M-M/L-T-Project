from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load the previously generated training and validation features.
    artifacts = joblib.load(DATA_DIR / "model_features.joblib")

    X_train = artifacts["X_train"]
    X_valid = artifacts["X_valid"]
    y_train = np.asarray(artifacts["y_train"])
    y_valid = np.asarray(artifacts["y_valid"])

    # Use the hyperparameters selected during tuning.
    model = LGBMRegressor(
        colsample_bytree=0.9165996316800473,
        learning_rate=0.09508421672126002,
        max_depth=8,
        min_child_samples=83,
        n_estimators=476,
        num_leaves=52,
        reg_lambda=1.5212112147976886,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, predictions)
    rmse = np.sqrt(mean_squared_error(y_valid, predictions))

    print(f"Validation MAE (log-price):  {mae:.4f}")
    print(f"Validation RMSE (log-price): {rmse:.4f}")

    joblib.dump(
        {
            "model": model,
            "validation_mae": mae,
            "validation_rmse": rmse,
        },
        MODEL_DIR / "tuned_lightgbm.joblib",
    )

    print("Model saved successfully.")


if __name__ == "__main__":
    main()
