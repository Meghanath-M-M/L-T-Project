from pathlib import Path

import joblib
import pandas as pd

from preprocessing import FEATURE_COLUMNS


class MercariFeatureBuilder:
    """Transform raw listings into the model's 212 input features."""

    def __init__(self, artifact_dir=None):
        if artifact_dir is None:
            artifact_dir = Path(__file__).resolve().parents[1] / "data"
        else:
            artifact_dir = Path(artifact_dir)

        self.preprocessor = joblib.load(
            artifact_dir / "preprocessor.joblib"
        )
        self.text_pipeline = joblib.load(
            artifact_dir / "text_pipeline.joblib"
        )

        self.feature_names = (
            list(FEATURE_COLUMNS)
            + [f"name_svd_{i}" for i in range(100)]
            + [f"description_svd_{i}" for i in range(100)]
        )

    def transform(self, listings):
        """Convert raw listing data into 212 model features."""

        if isinstance(listings, dict):
            listings = pd.DataFrame([listings])
        elif not isinstance(listings, pd.DataFrame):
            listings = pd.DataFrame(listings)

        required = [
            "name",
            "item_condition_id",
            "category_name",
            "brand_name",
            "shipping",
            "item_description",
        ]

        missing = [
            col for col in required
            if col not in listings.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        processed = self.preprocessor.transform(listings.copy())

        name_vectorizer = self.text_pipeline["name_vectorizer"]
        description_vectorizer = self.text_pipeline[
            "description_vectorizer"
        ]
        name_svd = self.text_pipeline["name_svd"]
        description_svd = self.text_pipeline["description_svd"]

        name_matrix = name_svd.transform(
            name_vectorizer.transform(processed["name_cleaned"])
        )

        description_matrix = description_svd.transform(
            description_vectorizer.transform(
                processed["item_description_cleaned"]
            )
        )

        structured = processed[FEATURE_COLUMNS].reset_index(drop=True)

        name_features = pd.DataFrame(
            name_matrix,
            columns=[f"name_svd_{i}" for i in range(100)],
        )

        description_features = pd.DataFrame(
            description_matrix,
            columns=[f"description_svd_{i}" for i in range(100)],
        )

        X = pd.concat(
            [structured, name_features, description_features],
            axis=1,
        )

        return X[self.feature_names]
