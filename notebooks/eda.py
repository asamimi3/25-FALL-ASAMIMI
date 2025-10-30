# jupytext metadata

# A. Imports & paths
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

sns.set_theme(style="whitegrid")

def find_project_root(start: Path = Path.cwd()) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").exists():
            return p
    raise RuntimeError("Couldn't locate project root (looked for a 'data' folder).")

def main():
    # Paths
    ROOT = find_project_root()
    DATA_RAW = ROOT / "data" / "raw"
    DATA_PROCESSED = ROOT / "data" / "processed"
    DOCS = ROOT / "docs"
    FIGURES = ROOT / "report" / "figures"

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    pd.options.display.max_rows = 25
    pd.options.display.max_columns = 60
    print("ROOT:", ROOT)
    print("RAW :", DATA_RAW)
    print("PROC:", DATA_PROCESSED)

    # --- export helpers ---
    def save_fig(name: str, dpi: int = 220) -> None:
        plt.tight_layout()
        plt.savefig(FIGURES / f"{name}.png", dpi=dpi, bbox_inches="tight")
        plt.close()

    def save_csv(df: pd.DataFrame, name: str) -> None:
        df.to_csv(DOCS / f"{name}.csv", index=False)

    # B. Load data
    cdc            = pd.read_parquet(DATA_PROCESSED / "cdc_weekly_clean.parquet")
    cdc_disp       = pd.read_parquet(DATA_PROCESSED / "cdc_disparities_by_week.parquet")
    brfss20_eda    = pd.read_parquet(DATA_PROCESSED / "brfss2020_eda.parquet")
    brfss22_eda    = pd.read_parquet(DATA_PROCESSED / "brfss2022_eda.parquet")
    brfss20_ready  = pd.read_parquet(DATA_PROCESSED / "brfss2020_ready.parquet")
    brfss22_ready  = pd.read_parquet(DATA_PROCESSED / "brfss2022_ready.parquet")
    equity         = pd.read_parquet(DATA_PROCESSED / "equity_selected.parquet")

    print("CDC Weekly:", cdc.shape)
    print("CDC Disparities:", cdc_disp.shape)
    print("BRFSS 2020 EDA:", brfss20_eda.shape)
    print("BRFSS 2022 EDA:", brfss22_eda.shape)
    print("BRFSS 2020 READY:", brfss20_ready.shape)
    print("BRFSS 2022 READY:", brfss22_ready.shape)
    print("Equity:", equity.shape)
    print("RaceH in brfss20_eda?", "RaceH" in brfss20_eda.columns)
    print("RaceH in brfss20_ready?", "RaceH" in brfss20_ready.columns)

    def quick_overview(df: pd.DataFrame, name: str) -> None:
        print(f"\n{name}")
        try:
            display(df.head(3))
            display(df.describe(include="all").T.head(10))
        except Exception:
            print(df.head(3).to_string())
            print(df.describe(include="all").T.head(10).to_string())
        print("\nMissing values per column:")
        print(df.isna().sum().sort_values(ascending=False).head(10))

    quick_overview(cdc, "CDC Weekly Cases & Deaths")
    quick_overview(brfss20_eda, "BRFSS 2020 EDA Survey")
    quick_overview(brfss22_eda, "BRFSS 2022 EDA Survey")
    quick_overview(equity, "Equity Dataset")

    # C. Baseline model (creates fpr/tpr etc. used later)
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, roc_curve, accuracy_score, precision_recall_fscore_support
    )

    df_ready = brfss20_ready.copy()
    y = df_ready["HeartDisease"].astype(int)
    X = df_ready.drop(columns=["HeartDisease"])

    split_path = DATA_PROCESSED / "brfss2020_split.csv"
    if split_path.exists():
        split = pd.read_csv(split_path).set_index("row_id")["split"]
    else:
        split = pd.Series(index=X.index, data="train")
        split.loc[X.sample(frac=0.2, random_state=42).index] = "test"

    X_train, X_test = X[split == "train"], X[split == "test"]
    y_train, y_test = y[split == "train"], y[split == "test"]

    clf = LogisticRegression(max_iter=1000, n_jobs=None, class_weight="balanced")
    clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba_test)
    fpr, tpr, thr = roc_curve(y_test, proba_test)
    pred_test = (proba_test >= 0.5).astype(int)
    acc = accuracy_score(y_test, pred_test)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
    print(f"AUC: {auc:.3f} | Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f}")

    # ROC figure
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--", alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (BRFSS 2020 test)")
    save_fig("roc_curve_brfss2020")

    # D. Fairness metrics (creates `fair`)
    def group_metrics(df, score_col="score", y_col="y", thr=0.5) -> pd.DataFrame:
        out = []
        for g, sub in df.groupby("RaceH", dropna=False):
            sub = sub.dropna(subset=[score_col])
            if len(sub) < 50:
                continue
            y_true = sub[y_col].to_numpy()
            y_hat  = (sub[score_col].to_numpy() >= thr).astype(int)
            auc_g  = roc_auc_score(y_true, sub[score_col]) if len(np.unique(y_true)) > 1 else np.nan
            acc_g  = accuracy_score(y_true, y_hat)
            prec_g, rec_g, f1_g, _ = precision_recall_fscore_support(
                y_true, y_hat, average="binary", zero_division=0
            )
            out.append({
                "RaceH": g, "n": len(sub), "AUC": auc_g, "ACC": acc_g,
                "Prec": prec_g, "Rec": rec_g, "F1": f1_g
            })
        return pd.DataFrame(out).sort_values("n", ascending=False)

    # make aligned score table
    scores = np.full(len(X), np.nan)
    test_pos = np.where(split.values == "test")[0]
    scores[test_pos] = proba_test
    score_df = pd.DataFrame({
        "row_id": np.arange(len(X)),
        "split": split.values,
        "y": y.values.astype(int),
        "score": scores
    })
    score_df["RaceH"] = brfss20_eda["RaceH"].reset_index(drop=True)

    fair = group_metrics(score_df.query("split=='test'"))
    display(fair)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=fair.melt(id_vars=["RaceH", "n"], value_vars=["AUC", "ACC", "F1"]),
        x="RaceH", y="value", hue="variable"
    )
    plt.ylim(0, 1); plt.title("Per-Race Metrics (test)"); plt.ylabel("score")
    save_fig("fairness_metrics_by_race")

    # E. Exploratory visuals (CDC / BRFSS)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=cdc, x="end_of_week", y="death_rate", hue="race_h")
    plt.title("COVID-19 Death Rates Over Time by Race (CDC Weekly Data)")
    plt.xlabel("Week"); plt.ylabel("Death Rate per 100k")
    save_fig("cdc_deathrates_over_time")

    plt.figure(figsize=(8, 6))
    sns.barplot(data=brfss20_eda, x="RaceH", y="HeartDisease", estimator=np.mean)
    plt.title("Heart Disease Prevalence by Race (BRFSS 2020)")
    plt.ylabel("Mean Heart Disease Rate")
    save_fig("brfss2020_heart_disease_by_race")

    plt.figure(figsize=(8, 5))
    sns.histplot(data=brfss22_eda, x="BMI", bins=30, kde=True)
    plt.title("Distribution of BMI (BRFSS 2022)")
    save_fig("brfss2022_bmi_distribution")

    # --- Death-rate ratios figure (include NH_White true values) ---

    # Merge base CDC weekly death rates (for NH_White) with disparities ratios
    cdc_white = (
        cdc.query("race_h == 'NH_White'")
        .rename(columns={"death_rate": "death_rate_NH_White"})
        [["end_of_week", "age_group", "death_rate_NH_White"]]
    )

    # Take the ratio table and compute actual death rates for all other races
    disp = cdc_disp.copy()
    ratio_cols = [c for c in disp.columns if c.startswith("death_ratio_")]
    disp = disp.merge(cdc_white, on=["end_of_week", "age_group"], how="left")

    # Reconstruct absolute death rates for each group using ratio * NH_White
    for col in ratio_cols:
        group = col.replace("death_ratio_", "")
        disp[f"death_rate_{group}"] = disp[col] * disp["death_rate_NH_White"]

    # Collect all absolute death-rate columns for melting
    rate_cols = ["death_rate_NH_White"] + [f"death_rate_{col.replace('death_ratio_', '')}" for col in ratio_cols]
    keep_cols = ["end_of_week", "age_group"] + rate_cols
    disp_abs = disp[keep_cols]

    # Melt into long format
    disp_long = disp_abs.melt(
        id_vars=["end_of_week", "age_group"],
        var_name="metric",
        value_name="death_rate"
    )
    disp_long["group"] = disp_long["metric"].str.replace("death_rate_", "", regex=False)

    # Consistent race order
    order = ["NH_White", "NH_Black", "Hispanic", "Asian", "AIAN"]
    disp_long["group"] = pd.Categorical(disp_long["group"], categories=order, ordered=True)

    # Plot for target age group
    age_sel = "50 - 64 Years"
    plot_df = disp_long.query("age_group == @age_sel").dropna(subset=["death_rate"])

    plt.figure(figsize=(11, 6))
    sns.lineplot(data=plot_df, x="end_of_week", y="death_rate", hue="group")
    plt.title(f"COVID-19 Death Rates by Race ({age_sel}) — Reconstructed from Ratios")
    plt.ylabel("Death rate per 100,000")
    plt.xlabel("Week")
    plt.legend(title="Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    save_fig("cdc_deathrates_reconstructed")

    # Save reconstructed absolute death rates for Tableau
    disp_abs.to_csv(DOCS / "cdc_reconstructed_absolute_for_tableau.csv", index=False)
    print(" Exported: cdc_reconstructed_absolute_for_tableau.csv")


    # F. Exports for Tableau
    roc_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thr})
    roc_df["AUC"] = auc
    save_csv(roc_df, "roc_for_tableau")

    fair_out = fair[["RaceH", "AUC", "ACC", "F1"]].sort_values("RaceH").round(3)
    save_csv(fair_out, "equity_for_tableau")

    save_csv(cdc, "cdc_weekly_for_tableau")
    save_csv(cdc_disp, "cdc_disparities_for_tableau")
    save_csv(brfss20_eda, "brfss2020_eda_for_tableau")
    save_csv(brfss22_eda, "brfss2022_eda_for_tableau")
    save_csv(equity, "equity_sdohtable_for_tableau")

    print("CSVs exported to:", DOCS)
    print("Figures saved to:", FIGURES)

# --- entrypoint ---
if __name__ == "__main__":
    main()
