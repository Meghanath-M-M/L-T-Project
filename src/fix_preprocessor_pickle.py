"""One-off fix: re-dump preprocessor.joblib with a clean module reference.

The original artifact was written by `python preprocessing.py`, which pickles
the class as __main__.MercariPreprocessor and breaks joblib.load() anywhere
except consumers that patch __main__ first. Re-pickling the loaded instance
records its real class (preprocessing.MercariPreprocessor) instead.
"""

import sys
from pathlib import Path

import __main__
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import MercariPreprocessor

# Allow unpickling the old __main__-style artifact
__main__.MercariPreprocessor = MercariPreprocessor

path = ROOT / "data" / "preprocessor.joblib"
prep = joblib.load(path)
joblib.dump(prep, path, compress=3)

# Verify it now loads without any __main__ patching
for attr in list(vars(__main__)):
    if attr == "MercariPreprocessor":
        delattr(__main__, attr)
reloaded = joblib.load(path)
print("Re-dumped", path)
print("Loads cleanly as:", type(reloaded).__module__ + "." + type(reloaded).__name__)
