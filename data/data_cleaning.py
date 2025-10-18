from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

#paths
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

#helpers
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
    df.to_parquet(DATA_PROCESSED / name, index=False)
    print(f"✓ Saved {name} ({len(df):,} rows)")


#-----------------------------------------
#CDC weekly demographics
def clean_cdc_weekly():
    f = next((p for p in DATA_RAW.glob("COVID-19_Weekly_Cases*.csv")), None)
    if f is None:
        raise FileNotFoundError("CDC file not found in data/raw (COVID-19_Weekly_Cases*.csv)")
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
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.query("jurisdiction == 'US' and sex == 'Overall'").copy()
    df.loc[:, "race_h"] = df["race_ethnicity"].map(RACE_MAP).fillna(df["race_ethnicity"])
    df = df[df["race_h"] != "Overall"]


    #save tidy
    tidy = df[["end_of_week", "age_group", "race_h", "case_rate", "death_rate"]].copy()
    save_parquet(tidy, "cdc_weekly_clean.parquet")


    #5disparties vs NH_White weekxage
    base = tidy.pivot_table(index=["end_of_week", "age_group"],
                            columns="race_h",
                            values="death_rate").reset_index()
    if "NH_White" in base.columns:
        for r in ["NH_Black", "Hispanic", "Asian", "AIAN", "Other/Multiple"]:
            if r in base.columns:
                base[f"death_gap_abs_{r}"] = base[r] - base["NH_White"]
                base[f"death_ratio_{r}"] = base[r] / base["NH_White"]

    disp = base.filter(regex="end_of_week|age_group|death_").sort_values(["end_of_week", "age_group"])
    save_parquet(disp, "cdc_disparities_by_week.parquet")


#---------------------------------
#Clean BRFSS 2020
from sklearn.model_selection import train_test_split

def clean_brfss_2020():
        f = DATA_RAW / "heart_2020_cleaned.csv"
        if not f.exists():
            raise FileNotFoundError("heart_2020_cleaned.csv not found in data/raw")
        df = pd.read_csv(f)

        #binary
        bin_cols = ["HeartDisease", "Smoking", "AlcoholDrinking", "Stroke",
                    "DiffWalking", "PhysicalActivity", "Asthma",
                    "KidneyDisease", "SkinCancer"]
        for c in bin_cols:
            if c in df.columns: df[c] = yesno_to_int(df[c])

        #ordinal GenHealth
        if "GenHealth" in df.columns:
            map_gh = {"Poor":1, "Fair":2, "Good":3, "Very good":4, "Excellent":5}
            df["GenHealthOrd"] = df["GenHealth"].map(map_gh).astype("Int64")

        #numeric
        for c in ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                df[c] = winsorize(df[c])

        #harmonize race/age
        if "Race" in df.columns:
            df["RaceH"] = df["Race"].map(RACE_MAP).fillna("Other/Multiple")
        if "AgeCategory" in df.columns:
            df["AgeCat"] = df["AgeCategory"].astype("category")

        #target & minimal drop of rows with missing target
        if "HeartDisease" not in df.columns:
            raise ValueError("Target HeartDisease not found in BRFSS 2020 file")
        df = df.dropna(subset=["HeartDisease"])

        #impute for remaining numeric NA
        for c in ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime", "GenHealthOrd"]:
            if c in df.columns:
                df[c] = df[c].fillna(df[c].median())

        # clean feature set
        keep = ["HeartDisease", "BMI","PhysicalHealth","MentalHealth","SleepTime",
                "Smoking","AlcoholDrinking","Stroke","DiffWalking",
                "PhysicalActivity","Asthma","KidneyDisease","SkinCancer",
                "GenHealthOrd","Sex","AgeCat","RaceH","Diabetic"]
        keep = [c for c in keep if c in df.columns]
        X = df[keep].copy()

        #EDA version before dummy encoding
        eda_cols = [
            "HeartDisease", "BMI", "PhysicalHealth", "MentalHealth", "SleepTime",
            "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity",
            "Asthma", "KidneyDisease", "SkinCancer", "GenHealthOrd", "Sex", "AgeCat", "RaceH", "Diabetic"
        ]

        # only keep the columns that actually exist in this df
        eda_cols = [c for c in eda_cols if c in df.columns]

        save_parquet(df[eda_cols], "brfss2020_eda.parquet")


        #one-hot encode small categoricals
        X = pd.get_dummies(X, columns=[c for c in ["Sex","AgeCat","RaceH","Diabetic"] if c in X.columns],
                           drop_first=True)

        save_parquet(X, "brfss2020_ready.parquet")

        #stratified spit labels for reproductibility
        if "RaceH" in df.columns:
            strat = pd.concat([df["HeartDisease"], df["RaceH"]], axis=1).astype(str).sum(axis=1)
            tr, te = train_test_split(X.index, test_size=0.2, random_state=42, stratify=strat)
            pd.Series(index=X.index, data=np.where(X.index.isin(tr), "train", "test")).to_csv(
                DATA_PROCESSED/"brfss2020_split.csv", index_label="row_id", header=["split"])


#---------------------------------
#clean BRFSS 2022
def clean_brfss_2022():
    #'with_nans' to demonstrate imputation; fall back otherwise
    f = DATA_RAW / "heart_2022_with_nans.csv"
    if not f.exists():
        f = DATA_RAW / "heart_2022_no_nans.csv"
    df = pd.read_csv(f)

    #Target
    target = "HadHeartAttack" if "HadHeartAttack" in df.columns else None
    if target is None:
        raise ValueError("Could not find HadHeartAttack in 2022 file")

    #Harmonize race/age
    rc = "RaceEthnicityCategory" if "RaceEthnicityCategory" in df.columns else "Race"
    df["RaceH"] = df[rc].map(RACE_MAP).fillna("Other/Multiple")

    ac = "AgeCategory" if "AgeCategory" in df.columns else None
    if ac: df["AgeCat"] = df[ac].str.replace("Age ","", regex=False).astype("category")

    #Binary fields
    for c in ["HadHeartAttack","HadAngina","HadStroke","PhysicalActivity","Asthma",
              "KidneyDisease","SkinCancer"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = yesno_to_int(df[c])

    #Numerics
    for c in ["BMI","PhysicalHealth","MentalHealth","SleepTime"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = winsorize(df[c])

    #impute simple
    for c in df.columns:
        if df[c].dtype.kind in "ifu":
            df[c] = df[c].fillna(df[c].median())
        elif df[c].dtype == object:
            df[c] = df[c].fillna("Unknown")

    # feature set similar to 2020
    keep = [target, "BMI","PhysicalHealth","MentalHealth","SleepTime",
            "PhysicalActivity","Asthma","KidneyDisease","SkinCancer",
            "Sex","AgeCat","RaceH"]
    keep = [c for c in keep if c in df.columns]
    X = df[keep].copy()

    # EDA versions -- RaceH as a single column
    brfss20_eda = pd.read_parquet(DATA_PROCESSED / "brfss2020_eda.parquet")
    brfss22_eda = pd.read_parquet(DATA_PROCESSED / "brfss2022_eda.parquet")

    # READY versions
    brfss20_ready = pd.read_parquet(DATA_PROCESSED / "brfss2020_ready.parquet")
    brfss22_ready = pd.read_parquet(DATA_PROCESSED / "brfss2022_ready.parquet")

    eda_cols = [
        "HadHeartAttack", "BMI", "PhysicalHealth", "MentalHealth", "SleepTime",
        "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking", "PhysicalActivity",
        "Asthma", "KidneyDisease", "SkinCancer", "GenHealthOrd", "Sex", "AgeCat", "RaceH", "Diabetic"
    ]
    eda_cols = [c for c in eda_cols if c in df.columns]
    save_parquet(df[eda_cols], "brfss2022_eda.parquet")


    X = pd.get_dummies(X, columns=[c for c in ["Sex","AgeCat","RaceH"] if c in X.columns],
                       drop_first=True)

    save_parquet(X, "brfss2022_ready.parquet")

#----------------------------
#Equity dataset
def clean_equity_subset():
    train = DATA_RAW / "train_clean.csv"
    test  = DATA_RAW / "test_clean.csv"
    frames = []

    # helper: tolerant, case-insensitive exact-name finder
    def find_col(df, targets):
        targets_l = {t.lower() for t in targets}
        for c in df.columns:
            if c.lower() in targets_l:
                return c
        return None

    for f in (train, test):
        if not f.exists():
            continue

        d = pd.read_csv(f, low_memory=False)
        print(f"{f.name} columns sample:", list(d.columns)[:40])

        # exact cols - if present
        keep_like = [
            "patient_age", "patient_race", "patient_state",
            "patient_zip3", "bmi", "payer_type"
        ]

        # many SES/env cols share prefixes; match case-insensitively
        ses_prefixes = (
            "income_", "education_", "poverty", "unemployment_rate",
            "limited_english", "health_uninsured", "rent_", "home_",
            "pm25", "ozone", "no2", "wealth_index", "hh_income_ratio", "education_ratio"
        )

        # keep exact matches…
        cols = [c for c in d.columns if c in keep_like]
        # …plus anything that STARTS WITH our prefixes (case-insensitive)
        cols += [c for c in d.columns if c.lower().startswith(ses_prefixes)]

        # optional diagnostic period column - spellings
        diag_col = find_col(d, ["DiagPeriodL90D", "DiagperiodL90D", "diagperiodl90d"])
        if diag_col:
            cols.append(diag_col)

        # de-dup just in case
        cols = list(dict.fromkeys(cols))

        if not cols:
            print(f"⚠ No matching columns found in {f.name}; skipping.")
            continue

        frames.append(d[cols].copy())

    if frames:
        out = pd.concat(frames, ignore_index=True)

        # harmonize race if available
        if "patient_race" in out.columns:
            out["RaceH"] = out["patient_race"].map(RACE_MAP).fillna("Other/Multiple")

        save_parquet(out, "equity_selected.parquet")
        print(f"Saved equity_selected.parquet ({len(out):,} rows)")
    else:
        print("No equity CSVs found or no matching columns; skipped.")


#----------------------
#entrypoint
if __name__ == "__main__":
    clean_cdc_weekly()
    clean_brfss_2020()
    clean_brfss_2022()
    clean_equity_subset()
