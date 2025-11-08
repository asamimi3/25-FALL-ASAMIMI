# predictivemodel

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# default style
sns.set_theme(style="whitegrid")

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DOCS = ROOT / "docs"
FIGURES = ROOT / "report" / "figures"

DOCS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

#data
df = pd.read_parquet(DATA_PROCESSED / "brfss2020_ready.parquet")

y = df["HeartDisease"].astype(int)
X = df.drop(columns=["HeartDisease"])

# Train/test split (hold test set for final evaluation only)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Base models ---
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
}

# --- Baseline performance (no tuning) ---
baseline_results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    baseline_results.append({
        "Model": name,
        "AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, preds),
        "F1": f1_score(y_test, preds),
    })

baseline_df = pd.DataFrame(baseline_results)
baseline_df.to_csv(DOCS / "model_comparison_baseline.csv", index=False)
print("\nBaseline performance:")
print(baseline_df)

# --- Hyperparameter tuning with cross-validation ---
param_grid = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "RandomForest": {
        "n_estimators": [100, 200],
        "max_depth": [5, 10, 20],
        "min_samples_split": [2, 5],
    },
    "GradientBoosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [2, 3, 5],
    },
}

best_models = {}
for name, model in models.items():
    print(f"\nRunning GridSearchCV for {name}...")
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid[name],
        cv=5,              # 5-fold cross-validation
        scoring="f1",      # optimize F1 because of class imbalance
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f"{name}: Best params -> {grid.best_params_}")

# --- Evaluate tuned models on the held-out test set ---
tuned_results = []
for name, model in best_models.items():
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    tuned_results.append({
        "Model": name,
        "AUC": roc_auc_score(y_test, proba),
        "Accuracy": accuracy_score(y_test, preds),
        "F1": f1_score(y_test, preds),
    })

tuned_df = pd.DataFrame(tuned_results)
tuned_df.to_csv(DOCS / "model_comparison_tuned.csv", index=False)

print("\nTuned performance (after GridSearchCV):")
print(tuned_df)

# --- Plot tuned model comparison for the report / Tableau ---
plt.figure(figsize=(8, 5))
plot_df = tuned_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric")
plt.title("Tuned Model Performance Comparison (BRFSS 2020)")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(FIGURES / "model_comparison_tuned.png", dpi=220)
plt.close()
