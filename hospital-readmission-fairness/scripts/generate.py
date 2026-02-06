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



if __name__ == "__main__":
    main()
