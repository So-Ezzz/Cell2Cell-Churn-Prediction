from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
RESULT_DIR = BASE_DIR / "results"

RANDOM_STATE = 42
TARGET_COL = "Churn"