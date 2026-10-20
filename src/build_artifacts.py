# ============================================================
# build_artifacts.py - Rebuild the ML artifacts from processed data
# Mercari Price Suggestion Challenge
#
# Produces:
#   data/text_pipeline.joblib     - fitted TF-IDF + TruncatedSVD models
#   data/model_features.joblib    - 212-column feature matrices for train.py
#
# Run:  python build_artifacts.py     (from the src/ directory)
# ============================================================

import gc
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import FEATURE_COLUMNS, TARGET  # noqa: E402

# Must match notebooks/02_Preprocessing_Feature_Engineering.ipynb
NAME_MAX_FEATURES = 10000
DESC_MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
N_COMPONENTS = 100
RANDOM_SEED = 42


def svd_features(svd, matrix):
    """Project a sparse TF-IDF matrix down to N_COMPONENTS float32 features."""
    dense = svd.transform(matrix)
    out = pd.DataFrame(dense, columns=range(N_COMPONENTS)).astype("float32")
    del dense
    gc.collect()
    return out


def main():
    started = time.time()
    needed = FEATURE_COLUMNS + ["name_cleaned", "item_description_cleaned", TARGET]
    raw_cols = [
        "name_cleaned",
        "item_description_cleaned",
    ]

    print("Loading processed splits (only the columns we need)...")
    train_df = pd.read_csv(DATA_DIR / "train_processed.csv", usecols=needed)
    valid_df = pd.read_csv(DATA_DIR / "valid_processed.csv", usecols=needed)
    print(f"  train={train_df.shape[0]:,} rows  valid={valid_df.shape[0]:,} rows")

    # TF-IDF treats missing text as a vocabulary token, so normalise first
    for df in (train_df, valid_df):
        for col in raw_cols:
            df[col] = df[col].fillna("")

    # ---- name text pipeline -------------------------------------
    print("Fitting name TF-IDF + SVD...")
    name_vectorizer = TfidfVectorizer(
        max_features=NAME_MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        dtype="float32",
    )
    name_train = name_vectorizer.fit_transform(train_df["name_cleaned"])
    name_valid = name_vectorizer.transform(valid_df["name_cleaned"])
    name_svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    name_svd.fit(name_train)
    name_train_feats = svd_features(name_svd, name_train)
    name_valid_feats = svd_features(name_svd, name_valid)
    del name_train, name_valid
    gc.collect()

    # ---- description text pipeline ------------------------------
    print("Fitting description TF-IDF + SVD...")
    description_vectorizer = TfidfVectorizer(
        max_features=DESC_MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=2,
        max_df=0.95,
        dtype="float32",
    )
    desc_train = description_vectorizer.fit_transform(
        train_df["item_description_cleaned"]
    )
    desc_valid = description_vectorizer.transform(
        valid_df["item_description_cleaned"]
    )
    description_svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=RANDOM_SEED)
    description_svd.fit(desc_train)
    desc_train_feats = svd_features(description_svd, desc_train)
    desc_valid_feats = svd_features(description_svd, desc_valid)
    del desc_train, desc_valid
    gc.collect()

    # ---- save the text pipeline artifact ------------------------
    text_pipeline = {
        "name_vectorizer": name_vectorizer,
        "description_vectorizer": description_vectorizer,
        "name_svd": name_svd,
        "description_svd": description_svd,
    }
    joblib.dump(text_pipeline, DATA_DIR / "text_pipeline.joblib", compress=3)
    print(f"Saved {DATA_DIR / 'text_pipeline.joblib'}")

    # ---- assemble the 212-column model matrices -----------------
    # Same order the app's MercariFeatureBuilder rebuilds at inference time
    feature_names = (
        list(FEATURE_COLUMNS)
        + [f"name_svd_{i}" for i in range(N_COMPONENTS)]
        + [f"description_svd_{i}" for i in range(N_COMPONENTS)]
    )

    print("Assembling model-ready feature matrices...")
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    name_train_feats.columns = [f"name_svd_{i}" for i in range(N_COMPONENTS)]
    name_valid_feats.columns = [f"name_svd_{i}" for i in range(N_COMPONENTS)]
    desc_train_feats.columns = [f"description_svd_{i}" for i in range(N_COMPONENTS)]
    desc_valid_feats.columns = [f"description_svd_{i}" for i in range(N_COMPONENTS)]

    X_train = pd.concat(
        [train_df[FEATURE_COLUMNS], name_train_feats, desc_train_feats], axis=1
    )[feature_names].astype("float32")
    X_valid = pd.concat(
        [valid_df[FEATURE_COLUMNS], name_valid_feats, desc_valid_feats], axis=1
    )[feature_names].astype("float32")

    y_train = train_df[TARGET].to_numpy(dtype="float32")
    y_valid = valid_df[TARGET].to_numpy(dtype="float32")

    assert not X_train.isna().any().any(), "NaNs in X_train"
    assert not X_valid.isna().any().any(), "NaNs in X_valid"
    print(f"  X_train={X_train.shape}  X_valid={X_valid.shape}")

    joblib.dump(
        {"X_train": X_train, "X_valid": X_valid, "y_train": y_train, "y_valid": y_valid},
        DATA_DIR / "model_features.joblib",
        compress=3,
    )
    print(f"Saved {DATA_DIR / 'model_features.joblib'}")
    print(f"Done in {(time.time() - started) / 60:.1f} minutes.")


if __name__ == "__main__":
    main()
