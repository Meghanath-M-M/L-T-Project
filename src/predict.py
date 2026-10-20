from pathlib import Path

import __main__
import joblib
import numpy as np

from features import MercariFeatureBuilder
from preprocessing import MercariPreprocessor

# Safety net: artifacts written by running preprocessing.py directly may
# reference __main__.MercariPreprocessor (same fix as app/app.py).
__main__.MercariPreprocessor = MercariPreprocessor


class MercariPricePredictor:
    """Predict prices from raw Mercari listing data."""

    def __init__(self, artifact_dir=None):
        if artifact_dir is None:
            artifact_dir = Path(__file__).resolve().parents[1] / "data"
        else:
            artifact_dir = Path(artifact_dir)

        self.feature_builder = MercariFeatureBuilder(artifact_dir)

        saved_model = joblib.load(
            artifact_dir / "models" / "tuned_lightgbm.joblib"
        )
        self.model = saved_model["model"]

    def predict(self, listings):
        """Return predicted prices in the original price scale."""

        X = self.feature_builder.transform(listings)

        predicted_log_prices = self.model.predict(X)
        predicted_prices = np.expm1(predicted_log_prices)

        return np.maximum(predicted_prices, 0)


if __name__ == "__main__":
    predictor = MercariPricePredictor()

    sample_listing = {
        "name": "Nike running shoes",
        "item_condition_id": 2,
        "category_name": "Women/Shoes/Athletic",
        "brand_name": "Nike",
        "item_description": "Lightly used running shoes in good condition",
        "shipping": 0,
    }

    price = predictor.predict(sample_listing)[0]
    print(f"Predicted price: ${price:.2f}")
