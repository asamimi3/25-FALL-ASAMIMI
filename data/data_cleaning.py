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

def save_parquet(df: pd.DataFrame, name:str):
    df.to_parquet(DATA_PROC / name, index=False)
    print(f"✓ Saved {name} ({len(df):,} rows)")


#CDC weekly demographics
#load
def clean_cdc_weekly():
    f = next((p for p in Data_RAW.glob("COVID-19_Weekly_Cases*.csv"), None)
    df = pd.read_csv(f, dtype=str)

    #normalize columns
    rename = {
        "End of Week": "end_of_week",
        "end_of_week": "end_of_week",
        "Jurisdiction": "jurisdiction",
        "Age Group": "age_group",
        "Sex": "sex",
        "Race and Ethnicity (Combined)": "race_ethnicity",
        "race_ethnicity_combined": "race_ethnicity",
        "Case crude rate (per 100,000)": "case_rate",
        "case_crude_rate_suppressed_per_100k": "case_rate",
        "Death crude rate(per 100,000)": "death_rate",
        "death_crude_rate_suppressed_per_100k": "death_rate",
    }

    df = df.rename(columns={k:v for k,v in rename.items() if k in df.columns})

    #types and filters
    df["end_of_week"] = pd.to_datetime(df["end_of_week"])
    for c in ("case_rate", "death_rate"):
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.query("jurisdiction == 'US' and sex == 'Overall'")
    df["race_h"] = df["race_ethnicity"].map(RACE_MAP).fillna(df["race_ethnicity"])
    df = df[df["race_h"] != "Overall"]


    #save tidy
    tidy = df[["end_of_week", "age_group", "race_h", "case_rate", "death_rate"]].copy()
    save_parquet(tidy, "cdc_weekly_clean.parquet")


    #5disparties vs NH_White weekxage
    base = tidy.pivot_table(index=["end_of_week", "age_group"],
                            columns="race_h", values="death_rate")
    base = base.reset_index()
    if "NH_White" in base.columns:
        for r in ["NH_Black","Hispanic","Asian","AIAN","Other/Multiple"]:
    if r in base.columns:
        base[f"death_gap_abs_{r}"] = base[r] - base["NH_White"]
        base[f"death_ratio_{r}"] = base[r] / base["NH_White"]

    disp = base.filter(regex="end_of_week|age_group|death_").copy()
    disp = disp.sort_values(["end_of_week", "age_group"])
    save_parquet(disp, "cddc_disparities_by_week.parquet")

