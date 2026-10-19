# ============================================================
# preprocessing.py - Data Preprocessing Pipeline
# Mercari Price Suggestion Challenge
# Member 1: Data Engineer (Data Understanding, EDA & Preprocessing)
#
# Hand-off to Member 2 (modelling):
#   from preprocessing import MercariPreprocessor, FEATURE_COLUMNS, TARGET
#   prep = joblib.load("../data/preprocessor.joblib")
#   X = prep.transform(raw_df)[FEATURE_COLUMNS]
# ============================================================

import re
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("default")

RANDOM_SEED = 42
UNKNOWN = "unknown"
TARGET = "log_price"

# Columns Member 2 should feed to a model (no raw text, no target, no ids)
FEATURE_COLUMNS = [
    "item_condition_id",
    "shipping",
    "brand_encoded",
    "category_1_encoded",
    "category_2_encoded",
    "category_3_encoded",
    "has_brand",
    "has_description",
    "name_length_log",
    "name_word_count_log",
    "description_length_log",
    "description_word_count_log",
]
# Cleaned text columns, for TF-IDF / embeddings
TEXT_COLUMNS = ["name_cleaned", "item_description_cleaned"]


def _clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return " ".join(text.split())


class MercariPreprocessor(BaseEstimator, TransformerMixin):
    """Missing values, text cleaning, feature engineering and encoding.

    fit() learns the label encoders from TRAIN data only; transform() applies
    exactly the same steps to any dataframe (train, validation or test).
    """

    def __init__(self):
        self.brand_encoder = LabelEncoder()
        self.category_encoders = {}
        self.is_fitted = False

    # ---------- fit / transform ----------
    def fit(self, X, y=None):
        # Reuse the transform steps so fit and transform can never disagree
        X = self._engineer_features(self._handle_missing_values(X.copy()))

        # 'unknown' is always added so unseen values can fall back to it
        self.brand_encoder.fit(
            pd.concat([X["brand_name"].astype(str), pd.Series([UNKNOWN])])
        )
        for col in ["category_1", "category_2", "category_3"]:
            le = LabelEncoder()
            le.fit(pd.concat([X[col], pd.Series([UNKNOWN])]))
            self.category_encoders[col] = le

        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform.")
        X = X.copy()
        X = self._handle_missing_values(X)
        X = self._clean_text_features(X)
        X = self._engineer_features(X)
        X = self._encode_categorical_features(X)
        X = self._log_transform_counts(X)
        return X

    # ---------- steps ----------
    def _handle_missing_values(self, X):
        X["brand_name"] = X["brand_name"].fillna(UNKNOWN).astype(str)
        X["category_name"] = X["category_name"].fillna(UNKNOWN).astype(str)
        X["item_description"] = X["item_description"].fillna("").astype(str)
        X["name"] = X["name"].fillna("").astype(str)
        return X

    def _clean_text_features(self, X):
        X["name_cleaned"] = X["name"].apply(_clean_text)
        X["item_description_cleaned"] = X["item_description"].apply(_clean_text)
        return X

    def _engineer_features(self, X):
        X["name_length"] = X["name"].str.len()
        X["name_word_count"] = X["name"].str.split().str.len()
        X["description_length"] = X["item_description"].str.len()
        X["description_word_count"] = X["item_description"].str.split().str.len()

        # category_name looks like "Women/Tops & Blouses/Blouse"
        parts = (
            X["category_name"].str.split("/", expand=True).reindex(columns=[0, 1, 2])
        )
        for i in range(3):
            X[f"category_{i + 1}"] = parts[i].fillna(UNKNOWN).replace("", UNKNOWN)

        X["has_brand"] = (X["brand_name"] != UNKNOWN).astype(int)
        X["has_description"] = (X["item_description"] != "").astype(int)
        return X

    def _encode_categorical_features(self, X):
        def safe_encode(series, encoder):
            known = set(encoder.classes_)
            return encoder.transform(series.where(series.isin(known), UNKNOWN))

        X["brand_encoded"] = safe_encode(X["brand_name"], self.brand_encoder)
        for col, enc in self.category_encoders.items():
            X[f"{col}_encoded"] = safe_encode(X[col], enc)
        return X

    def _log_transform_counts(self, X):
        for col in [
            "name_length",
            "name_word_count",
            "description_length",
            "description_word_count",
        ]:
            X[f"{col}_log"] = np.log1p(X[col])
        return X


# ============================================================
# Helper functions
# ============================================================


def preprocess_data(df, is_train=True, preprocessor=None):
    """Fit (if is_train) and apply the preprocessor. Returns (df, preprocessor)."""
    if preprocessor is None:
        if not is_train:
            raise ValueError("Pass a fitted preprocessor when is_train=False.")
        preprocessor = MercariPreprocessor()
    if is_train:
        preprocessor.fit(df)
    out = preprocessor.transform(df)
    print(f"Preprocessed: {df.shape} -> {out.shape}")
    return out, preprocessor


def handle_outliers(
    df, column="price", method="quantile", lower_quantile=0.01, upper_quantile=0.99
):
    """Cap or remove price outliers. Use on TRAINING data only.

    method: 'quantile' (cap), 'remove' (drop outside quantiles), 'iqr' (drop outside 1.5*IQR)
    """
    df = df.copy()
    n_before = len(df)
    if method == "quantile":
        lo, hi = df[column].quantile([lower_quantile, upper_quantile])
        df[column] = df[column].clip(lo, hi)
        print(f"Capped {column} to [{lo:.2f}, {hi:.2f}]")
    elif method in ("remove", "iqr"):
        if method == "iqr":
            q1, q3 = df[column].quantile([0.25, 0.75])
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        else:
            lo, hi = df[column].quantile([lower_quantile, upper_quantile])
        df = df[(df[column] >= lo) & (df[column] <= hi)]
        print(f"Removed {n_before - len(df):,} rows outside [{lo:.2f}, {hi:.2f}]")
    else:
        raise ValueError(f"Unknown method: {method}")
    return df


def log_transform_target(df, column="price"):
    df = df.copy()
    df[f"log_{column}"] = np.log1p(df[column])
    return df


# ============================================================
# Run as a script:  python preprocessing.py
# ============================================================
if __name__ == "__main__":
    import joblib
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("../data/train.tsv", sep="\t")
    print(f"Loaded {df.shape}")

    # Price == 0 listings are not useful for a price model
    zero = (df["price"] <= 0).sum()
    df = df[df["price"] > 0]
    print(f"Dropped {zero:,} rows with price <= 0")

    # Split FIRST so nothing learned from validation leaks into training
    train, valid = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    # Outlier capping on train only; validation stays untouched
    train = handle_outliers(train, "price", method="quantile")
    train = log_transform_target(train)
    valid = log_transform_target(valid)

    train_p, prep = preprocess_data(train, is_train=True)
    valid_p, _ = preprocess_data(valid, is_train=False, preprocessor=prep)

    train_p.to_csv("../data/train_processed.csv", index=False)
    valid_p.to_csv("../data/valid_processed.csv", index=False)
    joblib.dump(prep, "../data/preprocessor.joblib")
    print("Saved train_processed.csv, valid_processed.csv, preprocessor.joblib")
