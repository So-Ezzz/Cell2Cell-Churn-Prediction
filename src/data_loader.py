import pandas as pd
from src.config import RAW_DATA

def load_train():
    return pd.read_csv(RAW_DATA / "cell2celltrain.csv")

def load_holdout():
    return pd.read_csv(RAW_DATA / "cell2cellholdout.csv")