from pathlib import Path

import joblib
import lightgbm
import mlflow
import mlflow.lightgbm
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters selected by RandomizedSearchCV in
# notebooks/03_Modeling_Tuning.ipynb (3-fold CV, negative MAE scoring).
PARAMS = {
    "colsample_bytree": 0.9165996316800473,
    "learning_rate": 0.09508421672126002,
    "max_depth": 8,
    "min_child_samples": 83,
    "n_estimators": 476,
    "num_leaves": 52,
    "reg_lambda": 1.5212112147976886,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def main():
    # Load the previously generated training and validation features.
    artifacts = joblib.load(DATA_DIR / "model_features.joblib")

    X_train = artifacts["X_train"]
    X_valid = artifacts["X_valid"]
    y_train = np.asarray(artifacts["y_train"])
    y_valid = np.asarray(artifacts["y_valid"])

    # Track experiments in a SQLite registry at ./mlflow.db; run artifacts
    # (logged model, metrics files) go to ./mlruns
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("Mercari Price Suggestion")
    if experiment is None:
        experiment_id = client.create_experiment(
            "Mercari Price Suggestion",
            artifact_location=f"file:{ROOT / 'mlruns'}",
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        run_name="tuned-lightgbm", experiment_id=experiment_id
    ):
        model = LGBMRegressor(**PARAMS)
        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)

        mae = mean_absolute_error(y_valid, predictions)
        rmse = np.sqrt(mean_squared_error(y_valid, predictions))

        mlflow.log_params(PARAMS)
        mlflow.log_metrics(
            {
                "validation_mae": mae,
                "validation_rmse": rmse,
                "n_train_rows": len(X_train),
                "n_valid_rows": len(X_valid),
                "n_features": X_train.shape[1],
            }
        )
        mlflow.log_param("lightgbm_version", lightgbm.__version__)
        mlflow.lightgbm.log_model(model, name="tuned-lightgbm")

        joblib.dump(
            {
                "model": model,
                "validation_mae": mae,
                "validation_rmse": rmse,
            },
            MODEL_DIR / "tuned_lightgbm.joblib",
        )

        print(f"Validation MAE (log-price):  {mae:.4f}")
        print(f"Validation RMSE (log-price): {rmse:.4f}")
        print(f"MLflow run: {mlflow.active_run().info.run_id}")
        print("Model saved successfully.")


if __name__ == "__main__":
    main()
