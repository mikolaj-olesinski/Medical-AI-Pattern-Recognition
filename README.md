# Medical AI — Pattern Recognition & Fairness

Two case studies in clinical machine learning. Work in progress.

## [`ctg-classification/`](ctg-classification) — Fetal Heart-Rate Pattern Classification

Classic supervised classification pipeline on cardiotocography (CTG)
signals: EDA, preprocessing comparison, model selection across five
algorithms, feature importance mapped back to clinical meaning, and a
learning-curve check on generalization.

**Result:** Random Forest, 89.2% held-out accuracy. Top features
(accelerations, decelerations, variability) match standard obstetric
signals used by clinicians today.

## [`hospital-readmission-fairness/`](hospital-readmission-fairness) — Diabetic Readmission: Explainability & Fairness

Ensemble classification (Random Forest / LightGBM / XGBoost) on 100k+
hospital encounters, extended with **SHAP** explainability and a **fairness
audit** (equal opportunity by race/gender/age, plus a counterfactual check
for direct use of the race feature).
