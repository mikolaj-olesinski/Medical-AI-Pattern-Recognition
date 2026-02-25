"""
Regenerates figures for the readmission fairness case study.
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
FIGURES_DIR = "figures"


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/{name}", dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {FIGURES_DIR}/{name}")


def main():
    # ---- Load -------------------------------------------------------
    hosp = fetch_ucirepo(id=296)
    df = hosp.data.features.copy()
    df["readmitted"] = hosp.data.targets["readmitted"].values
    df["target"] = (df["readmitted"] == "<30").astype(int)
    print(f"Shape: {df.shape}")
    vc = df["target"].value_counts()
    print(f"Readmitted <30 days: {vc[1]} ({100 * vc[1] / len(df):.1f}%) of {len(df)}")

    # ---- EDA figures --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(["no <30-day\nreadmission", "<30-day\nreadmission"], vc.values,
                color=["#3498db", "#e74c3c"], alpha=0.85)
    for i, v in enumerate(vc.values):
        axes[0].text(i, v + 500, str(v), ha="center")
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("patients")

    rc = df["race"].value_counts()
    axes[1].bar(rc.index, rc.values, color=plt.cm.Set2(np.linspace(0, 1, len(rc))), alpha=0.85)
    axes[1].set_title("Race distribution")
    axes[1].set_xticklabels(rc.index, rotation=30, ha="right")

    gc = df["gender"].value_counts()
    axes[2].bar(gc.index, gc.values, color=["#9b59b6", "#2ecc71"], alpha=0.85)
    axes[2].set_title("Gender distribution")
    plt.suptitle("Hospital readmissions — demographics")
    savefig("demographics_overview.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col in zip(axes, ["race", "gender"]):
        rate = df.groupby(col)["target"].mean().sort_values(ascending=False)
        ax.bar(rate.index, rate.values, color=plt.cm.Set2(np.linspace(0, 1, len(rate))), alpha=0.85)
        for i, v in enumerate(rate.values):
            ax.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
        ax.set_title(f"Readmission rate by {col}")
        ax.set_xticklabels(rate.index, rotation=30, ha="right")
        ax.set_ylim(0, rate.max() * 1.3)
    plt.suptitle("Raw readmission rate by demographic group (before modeling)")
    savefig("readmission_rate_by_group.png")

    # ---- Preprocessing --------------------------------------------------
    sensitive_cols = ["race", "gender", "age"]
    drop_cols = ["readmitted", "target", "encounter_id", "patient_nbr",
                 "weight", "payer_code", "medical_specialty"]

    df_prep = df.replace("?", np.nan)
    sensitive = df_prep[sensitive_cols].copy()
    feat_cols = [c for c in df_prep.columns if c not in drop_cols]
    X = df_prep[feat_cols].copy()
    y = df["target"].values

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    X = X.values.astype(float)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    sens_test = sensitive.iloc[idx_test].reset_index(drop=True)
    print(f"Train: {X_train.shape}, test: {X_test.shape}, "
          f"features dropped for missingness/leakage: {drop_cols}")

    # ---- Models (fixed, literature-typical hyperparameters — no exhaustive sweep) ----
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "LightGBM": lgb.LGBMClassifier(
            num_leaves=63, learning_rate=0.05, min_child_samples=50,
            scale_pos_weight=pos_weight, verbose=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            max_depth=6, learning_rate=0.1, subsample=0.8,
            scale_pos_weight=pos_weight, verbosity=0, eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }
    for m in models.values():
        m.fit(X_train, y_train)

    # ---- Performance ------------------------------------------------
    rows = []
    for name, model in models.items():
        yp = model.predict(X_test)
        rows.append({
            "model": name,
            "recall": recall_score(y_test, yp),
            "precision": precision_score(y_test, yp),
            "f1": f1_score(y_test, yp),
            "balanced_accuracy": balanced_accuracy_score(y_test, yp),
        })
    perf = pd.DataFrame(rows).set_index("model")
    print("\nTest performance:")
    print(perf.round(4).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (name, model) in zip(axes, models.items()):
        yp = model.predict(X_test)
        cm = confusion_matrix(y_test, yp)
        ConfusionMatrixDisplay(cm, display_labels=["no <30", "<30"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nrecall={recall_score(y_test, yp):.3f}", fontsize=10)
    plt.suptitle("Confusion matrices — test set")
    savefig("confusion_matrices.png")



if __name__ == "__main__":
    main()
