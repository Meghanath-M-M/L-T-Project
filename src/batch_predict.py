# ============================================================
# batch_predict.py - Batch price inference over a CSV/TSV file
# Mercari Price Suggestion Challenge
#
# Usage:
#   python batch_predict.py --input ../data/valid_listings.csv --output ../data/predictions.csv
#
# The input file must contain the raw listing columns:
#   name, item_condition_id, category_name, brand_name, shipping,
#   item_description
# Extra columns (e.g. price) are ignored but carried through to the output,
# which makes it easy to score a hold-out file or check predictions against
# known prices.
# ============================================================

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import MercariPricePredictor  # noqa: E402

REQUIRED_COLUMNS = [
    "name",
    "item_condition_id",
    "category_name",
    "brand_name",
    "shipping",
    "item_description",
]

CHUNK_SIZE = 20_000


def main():
    parser = argparse.ArgumentParser(description="Predict prices in bulk.")
    parser.add_argument("--input", required=True, help="CSV or TSV of listings")
    parser.add_argument("--output", required=True, help="CSV for predictions")
    parser.add_argument(
        "--chunksize", type=int, default=CHUNK_SIZE, help="rows per chunk"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file not found: {input_path}")

    # train.tsv is tab-separated; everything else is treated as CSV
    sep = "\t" if input_path.suffix.lower() in {".tsv", ".tab"} else ","

    predictor = MercariPricePredictor()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    total = 0
    first_chunk = True

    # Stream the file so huge datasets never have to fit in memory at once
    for chunk in pd.read_csv(
        input_path, sep=sep, chunksize=args.chunksize, low_memory=False
    ):
        missing = [col for col in REQUIRED_COLUMNS if col not in chunk.columns]
        if missing:
            sys.exit(f"Input is missing required columns: {missing}")

        prices = predictor.predict(chunk[REQUIRED_COLUMNS])

        result = chunk.copy()
        result["predicted_price"] = np.round(prices, 2)
        result.to_csv(
            output_path, mode="w" if first_chunk else "a", index=False,
            header=first_chunk,
        )

        total += len(chunk)
        first_chunk = False
        print(f"  scored {total:,} rows...", end="\r")

    elapsed = time.time() - started
    print(f"\nScored {total:,} listings in {elapsed / 60:.1f} minutes.")
    print(f"Predictions written to {output_path}")


if __name__ == "__main__":
    main()
