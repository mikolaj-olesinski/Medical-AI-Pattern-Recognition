"""
Regenerates figures for the CTG case study.
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



if __name__ == "__main__":
    main()
