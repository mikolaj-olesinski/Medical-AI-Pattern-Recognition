"""
Regenerates all figures and prints the summary numbers used in the README
for the CTG fetal heart-rate pattern classification case study.

Run from the ctg-classification/ directory:
    python scripts/generate.py
"""
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
plt.style.use("default")
sns.set_palette("husl")

RANDOM_STATE = 42
FIGURES_DIR = "figures"

FEATURE_COLUMNS = [
    "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV",
    "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros",
    "Mode", "Mean", "Median", "Variance", "Tendency",
]
TARGET = "CLASS"

MEDICAL_MEANING = {
    "LB": "Baseline fetal heart rate",
    "AC": "Accelerations",
    "FM": "Fetal movements",
    "UC": "Uterine contractions",
    "DL": "Light decelerations",
    "DS": "Severe decelerations",
    "DP": "Prolonged decelerations",
    "ASTV": "Short-term variability (abnormal, %)",
    "MSTV": "Mean short-term variability",
    "ALTV": "Long-term variability (abnormal, %)",
    "MLTV": "Mean long-term variability",
    "Width": "FHR histogram width",
    "Min": "FHR histogram minimum",
    "Max": "FHR histogram maximum",
    "Nmax": "Number of histogram peaks",
    "Nzeros": "Number of histogram zeros",
    "Mode": "FHR histogram mode",
    "Mean": "FHR histogram mean",
    "Median": "FHR histogram median",
    "Variance": "FHR histogram variance",
    "Tendency": "Histogram tendency",
}


def savefig(name):
    path = f"{FIGURES_DIR}/{name}"
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved {path}")


def main():
    # ---- Load & clean -----------------------------------------------
    df = pd.read_csv("data/CTG.csv")
    df = df[FEATURE_COLUMNS + [TARGET]].apply(pd.to_numeric, errors="coerce")
    df[TARGET] = df[TARGET].astype("Int64")
    print(f"Rows: {len(df)} | features: {len(FEATURE_COLUMNS)} | classes: {df[TARGET].nunique()}")

    class_counts = df[TARGET].value_counts().sort_index()
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nClass distribution (%):")
    print((class_counts / len(df) * 100).round(2).to_string())
    print(f"\nMissing values: {'none' if missing.empty else missing.to_dict()}")

    # ---- EDA figure ----------------------------------------------------
    fig = plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    class_counts.plot(kind="bar", color="skyblue")
    plt.title("Class distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of recordings")
    plt.xticks(rotation=0)

    plt.subplot(2, 2, 2)
    df["LB"].hist(bins=30, alpha=0.75, color="seagreen")
    plt.title("Baseline FHR (LB) distribution")
    plt.xlabel("beats per minute")
    plt.ylabel("count")

    plt.subplot(2, 2, 3)
    df[["AC", "FM", "UC", "DL"]].boxplot()
    plt.title("Spread of key signal features")
    plt.xticks(rotation=30)

    plt.subplot(2, 2, 4)
    corr_subset = df[["LB", "AC", "FM", "UC", "ASTV", "MSTV", "Width", TARGET]].corr()
    sns.heatmap(corr_subset, annot=True, cmap="coolwarm", center=0, square=True,
                fmt=".2f", cbar_kws={"shrink": 0.8})
    plt.title("Correlation (selected features)")
    savefig("eda_overview.png")

    # ---- Preprocessing comparison --------------------------------------
    X = df[FEATURE_COLUMNS].dropna()
    y = df.loc[X.index, TARGET].astype(int)
    print(f"\nAfter dropping rows with missing values: {len(X)} samples "
          f"({len(df) - len(X)} removed)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    scaler_std = StandardScaler()
    X_std = pd.DataFrame(scaler_std.fit_transform(X), columns=X.columns, index=X.index)
    X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns, index=X.index)

    pca = PCA(n_components=10, random_state=RANDOM_STATE)
    X_pca = pd.DataFrame(pca.fit_transform(X_std), index=X.index)

    selector = SelectKBest(f_classif, k=10)
    X_sel = pd.DataFrame(selector.fit_transform(X, y),
                          columns=X.columns[selector.get_support()], index=X.index)

    data_versions = {
        "original": X, "standardized": X_std, "normalized": X_norm,
        "pca_10": X_pca, "selectk_10": X_sel,
    }
    print(f"PCA(10) explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    baseline = pd.Series(
        {name: cross_val_score(GaussianNB(), Xv, y, cv=skf).mean()
         for name, Xv in data_versions.items()},
        name="cv_accuracy",
    ).sort_values(ascending=False)
    print("\nBaseline (Naive Bayes) accuracy per data representation:")
    print(baseline.round(4).to_string())

    plt.figure(figsize=(7, 4))
    baseline.plot(kind="bar", color="#3498db", alpha=0.85)
    plt.ylabel("CV accuracy (Naive Bayes)")
    plt.title("Effect of preprocessing on a baseline classifier")
    plt.xticks(rotation=30, ha="right")
    for i, v in enumerate(baseline.values):
        plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    savefig("preprocessing_comparison.png")

    # ---- Model comparison -----------------------------------------------
    candidates = [
        ("Naive Bayes", GaussianNB(), "original"),
        ("Decision Tree (default)", DecisionTreeClassifier(random_state=RANDOM_STATE), "original"),
        ("Decision Tree (tuned)",
         DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=RANDOM_STATE),
         "original"),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE), "original"),
        ("SVM (RBF)", SVC(kernel="rbf", random_state=RANDOM_STATE), "pca_10"),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=RANDOM_STATE), "original"),
    ]

    rows = []
    for name, model, data_name in candidates:
        Xv = data_versions[data_name]
        cv_acc = cross_val_score(model, Xv, y, cv=skf, scoring="accuracy").mean()
        rows.append({"model": name, "data": data_name, "cv_accuracy": cv_acc})
    results = pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
    print("\nModel comparison (5-fold CV accuracy):")
    print(results.round(4).to_string(index=False))

    plt.figure(figsize=(8, 4.5))
    plt.barh(results["model"], results["cv_accuracy"], color="#2ecc71", alpha=0.85)
    plt.gca().invert_yaxis()
    plt.xlabel("CV accuracy")
    plt.title("Model comparison — CTG pattern classification")
    for i, v in enumerate(results["cv_accuracy"]):
        plt.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    savefig("model_comparison.png")

    # ---- Best model — detailed evaluation --------------------------------
    best_name, best_data = results.iloc[0][["model", "data"]]
    best_model = dict((n, m) for n, m, _ in candidates)[best_name]
    Xb = data_versions[best_data]

    X_train, X_test, y_train, y_test = train_test_split(
        Xb, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nBest model: {best_name} on '{best_data}' data")
    print(f"Held-out test accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_pred))

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[f"class {c}" for c in labels],
                yticklabels=[f"class {c}" for c in labels],
                cbar_kws={"label": "count"})
    plt.title(f"Confusion matrix — {best_name}\ntest accuracy: {test_acc:.3f}")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    savefig("confusion_matrix.png")

    # ---- Feature importance ---------------------------------------------
    if hasattr(best_model, "feature_importances_") and best_data == "original":
        importance = pd.Series(best_model.feature_importances_, index=X.columns)
        importance = importance.sort_values(ascending=False)
    else:
        # fall back to a freshly-trained tuned Decision Tree on original features
        # so the importance ranking stays interpretable in clinical terms
        dt = DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=RANDOM_STATE)
        dt.fit(X, y)
        importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)

    top10 = importance.head(10)
    plt.figure(figsize=(8, 5))
    plt.barh(top10.index[::-1], top10.values[::-1], color="#3498db", alpha=0.85)
    plt.xlabel("Feature importance")
    plt.title("Top 10 features — Decision Tree")
    for i, v in enumerate(top10.values[::-1]):
        plt.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    savefig("feature_importance.png")

    print("\nTop 5 features (clinical meaning):")
    for feat, val in importance.head(5).items():
        print(f"  {feat:8} {val:.3f}  {MEDICAL_MEANING.get(feat, '')}")
    print(f"Top 5 features account for {importance.head(5).sum()*100:.1f}% of total importance")

    # ---- Learning curve ----------------------------------------------
    dt_tuned = DecisionTreeClassifier(max_depth=8, min_samples_split=5, random_state=RANDOM_STATE)
    train_sizes, train_scores, val_scores = learning_curve(
        dt_tuned, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=RANDOM_STATE, scoring="accuracy",
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(train_sizes, train_mean, "o-", color="tab:blue", label="training set")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="tab:blue")
    plt.plot(train_sizes, val_mean, "o-", color="tab:red", label="validation")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="tab:red")
    plt.xlabel("training set size")
    plt.ylabel("accuracy")
    plt.title("Learning curve — Decision Tree (tuned)")
    plt.legend()
    plt.grid(alpha=0.3)
    gap = train_mean[-1] - val_mean[-1]
    print(f"\nLearning curve gap (train - val) at full size: {gap:.4f}")
    savefig("learning_curve.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
