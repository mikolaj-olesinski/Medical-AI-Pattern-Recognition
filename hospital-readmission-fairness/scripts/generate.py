"""
Regenerates all figures and prints the summary numbers used in the README
for the diabetic hospital readmission / fairness case study.

Downloads the "Diabetes 130-US hospitals" dataset from the UCI repository
(needs internet access on first run). Run from the
hospital-readmission-fairness/ directory:

    python scripts/generate.py
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

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(perf))
    width = 0.2
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for i, (metric, color) in enumerate(zip(perf.columns, colors)):
        bars = ax.bar(x + (i - 1.5) * width, perf[metric], width, label=metric, color=color, alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(perf.index)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model performance — early readmission prediction")
    ax.legend()
    savefig("performance_comparison.png")

    # ---- SHAP -------------------------------------------------------
    best_name = perf["f1"].idxmax()
    best_model = models[best_name]
    print(f"\nBest model by F1: {best_name}")

    explainer = shap.TreeExplainer(best_model)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(X_test.shape[0], min(2000, X_test.shape[0]), replace=False)
    X_shap = X_test[sample_idx]
    sv = explainer.shap_values(X_shap)
    if isinstance(sv, list):
        sv = sv[1]
    shap_exp = shap.Explanation(values=sv, data=X_shap, feature_names=feat_cols)

    plt.figure(figsize=(9, 7))
    shap.plots.beeswarm(shap_exp, max_display=15, show=False)
    plt.title(f"SHAP beeswarm — {best_name}")
    savefig("shap_beeswarm.png")

    plt.figure(figsize=(8, 5))
    shap.plots.bar(shap_exp, max_display=12, show=False)
    plt.title(f"SHAP mean |value| — top features ({best_name})")
    savefig("shap_feature_importance.png")

    mean_abs = np.abs(sv).mean(axis=0)
    top_feats = pd.Series(mean_abs, index=feat_cols).sort_values(ascending=False).head(5)
    print("Top 5 SHAP features:")
    print(top_feats.round(4).to_string())

    # ---- Fairness: equal opportunity ---------------------------------
    def recall_per_group(model, group_col, values):
        yp = model.predict(X_test)
        out = {}
        for g in values:
            mask = sens_test[group_col].values == g
            if mask.sum() == 0 or y_test[mask].sum() == 0:
                continue
            out[g] = recall_score(y_test[mask], yp[mask])
        return out

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fairness_summary = {}
    for ax, col in zip(axes, ["race", "gender"]):
        values = sorted(sens_test[col].dropna().unique())
        all_recalls = {name: recall_per_group(m, col, values) for name, m in models.items()}
        fairness_summary[col] = all_recalls
        width = 0.25
        x = np.arange(len(values))
        for i, (name, recalls) in enumerate(all_recalls.items()):
            vals = [recalls.get(g, 0) for g in values]
            ax.bar(x + (i - 1) * width, vals, width, label=name, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(values, rotation=30, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Recall (true positive rate)")
        ax.set_title(f"Equal opportunity — recall by {col}")
        ax.legend(fontsize=8)
    plt.suptitle("Fairness check: does recall differ across protected groups?")
    savefig("fairness_equal_opportunity.png")

    for col, recalls in fairness_summary.items():
        print(f"\nRecall by {col}:")
        for name, r in recalls.items():
            print(f"  {name}: " + ", ".join(f"{g}={v:.3f}" for g, v in r.items()))

    # ---- Counterfactual: does the model use race directly? --------------
    best_idx = feat_cols.index("race") if "race" in feat_cols else None
    if best_idx is not None:
        le_race = encoders["race"]
        caucasian_code = le_race.transform(["Caucasian"])[0]
        X_test_cf = X_test.copy()
        X_test_cf[:, best_idx] = caucasian_code

        print("\nCounterfactual: setting race=Caucasian for every test row")
        rows = []
        for name, model in models.items():
            yp_orig = model.predict(X_test)
            yp_cf = model.predict(X_test_cf)
            changed = (yp_orig != yp_cf).mean() * 100
            rows.append({"model": name, "predictions_changed_%": changed})
            print(f"  {name}: {changed:.2f}% of predictions changed")

        best_yp = best_model.predict(X_test)
        best_yp_cf = best_model.predict(X_test_cf)
        groups = sorted(sens_test["race"].dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, (yp, label) in zip(axes, [(best_yp, "original race"), (best_yp_cf, "all set to Caucasian")]):
            recalls = []
            for g in groups:
                mask = sens_test["race"].values == g
                recalls.append(recall_score(y_test[mask], yp[mask]) if mask.sum() and y_test[mask].sum() else 0)
            ax.bar(groups, recalls, color="#3498db", alpha=0.85)
            ax.set_title(f"{best_name} — {label}")
            ax.set_ylabel("Recall")
            ax.set_ylim(0, 1.0)
            ax.set_xticklabels(groups, rotation=30, ha="right")
        plt.suptitle("Counterfactual fairness check (race feature)")
        savefig("counterfactual_race.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
