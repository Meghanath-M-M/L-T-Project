# Mercari Price Suggestion — L&T Project

Machine-learning web app that suggests a marketplace price for a product listing
from its name, category, brand, condition, shipping and description.

## Team Members:Meghanath,Sehal,Sabeel
## How it works

| Stage | Artifact | Built by |
| --- | --- | --- |
| Structured preprocessing | `data/preprocessor.joblib` | `src/preprocessing.py` |
| Text features (TF-IDF + TruncatedSVD) | `data/text_pipeline.joblib` | `src/build_artifacts.py` |
| Tuned LightGBM regressor | `data/models/tuned_lightgbm.joblib` | `src/train.py` |

At inference time `src/features.py` (`MercariFeatureBuilder`) loads the two
feature artifacts and rebuilds the same 212-column matrix the model was trained
on (12 structured + 100 name SVD + 100 description SVD features);
`src/predict.py` wraps it into `MercariPricePredictor`.

## Repository layout

```
app/app.py                     Streamlit UI
src/
  preprocessing.py             MercariPreprocessor + FEATURE_COLUMNS
  build_artifacts.py           Fits TF-IDF + SVD, writes text_pipeline.joblib
  train.py                     Trains LightGBM on model_features.joblib
  features.py                  MercariFeatureBuilder (inference features)
  predict.py                   MercariPricePredictor (inference end-to-end)
notebooks/                     EDA, preprocessing, modelling, explainability
data/                          Artifacts (.joblib, tracked) + datasets (ignored)
```

The raw Kaggle datasets (`train.tsv`, the processed CSVs) are **not** stored in
GitHub — only the fitted `.joblib` artifacts are. The datasets are gitignored;
re-run the notebooks to regenerate them from the raw data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Rebuild the artifacts

Only needed if the data or feature logic changes. Requires the processed CSVs
in `data/` (produced by `notebooks/02_Preprocessing_Feature_Engineering.ipynb`).

```bash
python src/build_artifacts.py   # writes data/text_pipeline.joblib + data/model_features.joblib
python src/train.py             # writes data/models/tuned_lightgbm.joblib
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (the fitted `.joblib` files are already
   tracked — `data/model_features.joblib` and the CSVs stay ignored).
2. Open <https://share.streamlit.io> → **Create app** → pick this repo.
3. Set **Main file path** to `app/app.py` and click **Deploy**.
   The pinned versions in `requirements.txt` match the versions the artifacts
   were pickled with, so the model loads cleanly.
