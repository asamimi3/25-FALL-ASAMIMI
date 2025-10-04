from pathlib import Path
import pandas as pd
import numpy as np

#paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DATA_PROC.mkdir(parents=true, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

#harmonize
RACE_MAP = {
    "White": "NH_White", "White, NH": "NH_White",
    "Black": "NH_Black", "Black, NH": "NH_Black",
    "Hispanic": "Hispanic", "Hispanic/Latino": "Hispanic",
    "Asian": "Asian", "Asian/PI, NH": "Asian",
    "American Indian/Alaskan Native": "AIAN", "AI/AN, NH": "AIAN",
    "Other": "Other/Multiple", "Multiracial, Non-Hispanic": "Other/Multiple",
    "Other race only, Non-Hispanic": "Other/Multiple",
    "Unknown": "Unknown", "Overall": "Overall"
}

def yesno_to_int(series: pd.Series) -> pd.Series:
    return series.map({"Yes":1, "No":0}).astype("Int64")

def winsorize(s: pd.Series, lower=0.01, upper=0.99):
    ql, qu = s.quantile([lower, upper])
    return s.clip(ql, qu)
