# Mercari Price Suggestion — L&T Project

**Team members:** Meghanath, Sehal, Sabeel
## Live Demo Link:https://mercari-price-predictor.streamlit.app/
## Problem statement

Online marketplaces contain millions of listings with unstructured text
(product name, free-text description) mixed with structured attributes
(category, brand, condition, shipping). Sellers frequently misprice items.
This project builds a machine-learning system that suggests a price for a
listing given only its metadata and text, and serves it as a deployable
web application.

The problem is framed as regression on the log-transformed price,
`log_price = ln(1 + price)`, to reduce the skew of a target spanning
$0.01 to $2,000+. Predictions are converted back to dollars with
`price = exp(log_price) − 1` and clipped at zero.

## Dataset source and description

Source: [Kaggle — Mercari Price Suggestion Challenge](https://www.kaggle.com/c/mercari-price-suggestion-challenge).

The raw `train.tsv` (tab-separated, ~1.48 million listings) contains:

| Column | Description |
| --- | --- |
| `train_id` | Listing identifier (dropped before modelling) |
| `name` | Product title |
| `item_condition_id` | Seller-declared condition, 1–5 |
| `category_name` | Hierarchical category, e.g. `Women/Shoes/Athletic` |
| `brand_name` | Brand (many missing values) |
| `shipping` | 1 = seller pays, 0 = buyer pays |
| `item_description` | Free-text description (often contains boilerplate) |
| `price` | Target price in USD |

**The raw and processed datasets are intentionally not committed to GitHub**
(600 MB+). They are gitignored; see
[Obtaining the dataset](#obtaining-the-dataset) to recreate them.

## Installation instructions

Requires Python 3.11.

```bash
git clone https://github.com/Meghanath-M-M/L-T-Project.git
cd L-T-Project
pip install -r requirements.txt
```

All dependency versions are pinned to the versions the `.joblib` artifacts
were pickled with, so the saved model loads without retraining.

## Project structure

```
app/app.py                     Streamlit web UI (deployable entry point)
src/
  preprocessing.py             MercariPreprocessor: cleaning + 12 structured features
  build_artifacts.py           Fits TF-IDF + TruncatedSVD, writes text_pipeline.joblib
  train.py                     Trains tuned LightGBM, logs to MLflow, saves model
  features.py                  MercariFeatureBuilder: rebuilds the 212-feature matrix
  predict.py                   MercariPricePredictor: single-listing inference
  batch_predict.py             Batch inference over a CSV/TSV file
  fix_preprocessor_pickle.py   One-off artifact portability fix
notebooks/
  01_EDA.ipynb                 EDA and data-quality analysis
  02_Preprocessing_Feature_Engineering.ipynb
  03_Modeling_Tuning.ipynb     Model comparison + RandomizedSearchCV tuning
  04_Ensemble_Explainability.ipynb  Voting ensemble + SHAP analysis
data/                          .joblib artifacts (tracked) + datasets (gitignored)
mlruns/                        MLflow experiment-tracking registry (tracked)
```

## Obtaining the dataset

1. Download `train.tsv.zip` from the Kaggle competition page above
   (free account required; the license forbids redistribution, hence it is
   not in this repository).
2. Unzip and place `train.tsv` in `data/`.
3. Rebuild the derived data:

```bash
python src/preprocessing.py      # writes data/train_processed.csv, valid_processed.csv, preprocessor.joblib
python src/build_artifacts.py    # writes data/text_pipeline.joblib + data/model_features.joblib
python src/train.py              # writes data/models/tuned_lightgbm.joblib and logs to MLflow
```

No download is needed just to **run the app** — the fitted artifacts are
already committed.

## Data-preprocessing approach

Implemented in `src/preprocessing.py` (`MercariPreprocessor`, a sklearn
`BaseEstimator` so `fit`/`transform` can never disagree):

1. **Split first** — 80/20 train/validation split (`random_state=42`) before
   anything is fitted, so validation data can never leak into training.
2. **Missing values** — missing brand → `unknown`, missing category →
   `unknown`, missing description/name → empty string.
3. **Target handling** — listings with `price ≤ 0` dropped (~83k rows);
   prices capped at the 1st/99th percentile on the **training split only**;
   target transformed to `ln(1 + price)`.
4. **Text cleaning** — lowercased, stripped of punctuation, whitespace
   normalised, producing `name_cleaned` and `item_description_cleaned`.
5. **Encoding** — brand and the three category levels label-encoded from
   training data only; unseen values at inference fall back to `unknown`.

Result: `data/train_processed.csv` (1,185,328 rows) and
`data/valid_processed.csv` (296,333 rows), plus the fitted
`data/preprocessor.joblib`.

## Feature-engineering approach

Each listing becomes a **212-dimension** vector:

- **12 structured features** — condition, shipping, label-encoded brand and
  3 category levels, `has_brand`, `has_description`, and log-transformed
  length/word-count statistics for name and description.
- **100 name features** — TF-IDF (10k features, uni+bigrams, `min_df=2`)
  reduced by TruncatedSVD to 100 components.
- **100 description features** — TF-IDF (20k features, uni+bigrams,
  `min_df=2`, `max_df=0.95`) reduced by TruncatedSVD to 100 components.

TF-IDF and SVD are fitted on the **training split only** and applied to
validation data. The fitted objects live in `data/text_pipeline.joblib`.
At inference, `src/features.py` rebuilds the exact same 212-column matrix
from raw listings, keeping training and serving consistent.

## Models evaluated

All metrics below are on the held-out validation set in log-price space.
Tree/boosting experiments used a 100k-row training sample for practicality
on a 16 GB machine; evaluation always used the full validation set.

| Model | MAE (log) | RMSE (log) |
| --- | --- | --- |
| Baseline (`DummyRegressor`, mean) | 0.5817 | 0.7455 |
| Ridge regression (α=1.0) | 0.5643 | 0.7219 |
| Random Forest | 0.4535 | 0.5937 |
| LightGBM (default hyperparameters) | 0.4206 | 0.5519 |
| **LightGBM (tuned)** | **0.4109** | **0.5411** |
| Voting ensemble (Ridge + tuned LightGBM) | 0.4593 | 0.5959 |

## Hyperparameter-tuning approach

`RandomizedSearchCV` on LightGBM with **3-fold cross-validation**,
optimising negative MAE (6 random configurations, ~3.5 minutes):

| Hyperparameter | Search space | Best value |
| --- | --- | --- |
| `n_estimators` | 200–600 | 476 |
| `learning_rate` | 0.02–0.10 | 0.0951 |
| `num_leaves` | 20–60 | 52 |
| `max_depth` | −1, 8, 12, 16 | 8 |
| `min_child_samples` | 20–100 | 83 |
| `colsample_bytree` | 0.7–1.0 | 0.9166 |
| `reg_lambda` | 0.0–5.0 | 1.5212 |

The tuned configuration beat the default LightGBM (MAE 0.4206 → 0.4109).
A Ridge + LightGBM voting ensemble was also tried and performed **worse**
(0.4593) — averaging a weak linear model with the gradient booster diluted
its predictions, so the tuned LightGBM was kept as the final model.

## Final evaluation results

The final model (`src/train.py`) uses the tuned hyperparameters retrained
on the **full 1.18M-row training split**:

| Metric | Value |
| --- | --- |
| Validation MAE (log-price) | **0.3903** |
| Validation RMSE (log-price) | **0.5157** |

This beats the notebook's 100k-sample tuned result (0.4109), confirming the
model benefits from the full dataset. Saved as
`data/models/tuned_lightgbm.joblib` with metrics embedded in the dictionary.

## Explainability results

`notebooks/04_Ensemble_Explainability.ipynb` applies **SHAP**
(`shap.TreeExplainer`) to the LightGBM model on a 2,000-row validation
sample (kept small to avoid out-of-memory errors):

- The SHAP summary/bar plot ranks features by global impact on predicted
  price and is embedded in the notebook.
- The analysis contrasts structured signals (encoded brand, category,
  condition) against the SVD-derived text components, showing how much of
  the price signal comes from free text versus metadata.
- Per-listing SHAP values also explain individual predictions
  (additive contributions per feature in log-price space).

## MLflow tracking details

`src/train.py` logs every training run to the local MLflow registry in
`mlruns/` (committed to this repository as tracking evidence):

- **Experiment:** `Mercari Price Suggestion`, run name `tuned-lightgbm`
- **Logged parameters:** all seven tuned hyperparameters + seed settings
- **Logged metrics:** `validation_mae`, `validation_rmse`, plus dataset
  shape metrics (`n_train_rows`, `n_valid_rows`, `n_features`)
- **Logged artifact:** the trained LightGBM model (`tuned-lightgbm/`)
- Run metadata (versions, timestamps, run id) is stored under `mlruns/`

View the runs locally with:

```bash
mlflow ui --backend-store-uri ./mlruns
# then open http://localhost:5000
```

## Batch-inference instructions

Score any CSV/TSV of listings with `src/batch_predict.py`. The input must
contain `name`, `item_condition_id`, `category_name`, `brand_name`,
`shipping`, `item_description`; extra columns are carried through, and rows
are streamed in 20k chunks so large files never need to fit in memory.

```bash
# Score a tab-separated Mercari file
python src/batch_predict.py --input data/valid.tsv --output data/predictions.csv

# Score a CSV
python src/batch_predict.py --input listings.csv --output predictions.csv
```

The output CSV is the input plus a `predicted_price` column (USD).

## Local application instructions

```bash
streamlit run app/app.py
```

The app loads the preprocessor, text pipeline and tuned LightGBM from
`data/`, then offers a form (name, category, brand, condition, shipping,
description) that returns a suggested listing price. For a quick CLI check
of the same inference path:

```bash
python src/predict.py     # prints a predicted price for a sample listing
```

## Deployment URL

Streamlit Community Cloud: https://mercari-price-predictor.streamlit.app/

Deployment steps: push this repository to GitHub, open
<https://share.streamlit.io>, create an app pointing at
`Meghanath-M-M/L-T-Project` with main file path `app/app.py`, and deploy.
The pinned `requirements.txt` installs the exact dependency versions the
artifacts were pickled with, so the model loads directly from the committed
`.joblib` files — no dataset download required.

## Limitations and future improvements

**Limitations**

- TF-IDF + SVD gives linear, bag-of-words text features; it loses word order
  and semantics, so novel or misspelled listings fall back to weak signals.
- Brand/category encoding is label-based; unseen brands degrade to the
  `unknown` code rather than a learned embedding.
- Hyperparameter search was deliberately small (6 configurations, 3 folds)
  to stay within local compute limits — more budget would likely improve it.
- Price-capped training data (1st–99th percentile) means extreme luxury or
  bulk listings are systematically under-predicted.
- Explainability is computed on a 2,000-row sample, not the full dataset.

**Future improvements**

- Replace TF-IDF/SVD with a pretrained text encoder (e.g. sentence
  transformers) for semantic text features.
- Upgrade experiment tracking to a remote MLflow server or hosted tracker
  for team-wide comparison.
- Expand the tuning budget (Optuna with 100+ trials) and try
  CatBoost/XGBoost with native categorical support.
- Add a REST API (FastAPI) alongside the Streamlit UI for programmatic use.
- Add unit tests for the preprocessing/inference path and a CI workflow.
