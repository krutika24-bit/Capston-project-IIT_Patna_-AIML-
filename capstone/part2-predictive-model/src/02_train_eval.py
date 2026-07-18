"""
Train and evaluate late-delivery prediction models.

Loads data/model_dataset.csv, performs a TIME-BASED train/test split
(80% earliest orders = train, 20% latest = test), builds a sklearn
pipeline with ColumnTransformer, trains LogisticRegression and
RandomForestClassifier, evaluates with ROC-AUC, F1, classification_report,
confusion_matrix, prints top-10 RandomForest feature importances, and
saves the better model to model/model.pkl with metadata in
model/feature_list.json.

Usage: python src/02_train_eval.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PART2 = os.path.normpath(os.path.join(HERE, ".."))
DATA_CSV = os.path.join(PART2, "data", "model_dataset.csv")
MODEL_DIR = os.path.join(PART2, "model")
MODEL_PKL = os.path.join(MODEL_DIR, "model.pkl")
FEATURE_JSON = os.path.join(MODEL_DIR, "feature_list.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================================
# 1.  Load data
# ============================================================================
print("=" * 65)
print("LOADING DATA")
print("=" * 65)
df = pd.read_csv(DATA_CSV)
print(f"  Loaded: {DATA_CSV}")
print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"  Target balance:\n    late={df['target_late'].sum()}  on-time={(1-df['target_late']).sum()}")

# ============================================================================
# 2.  Time-based train / test split
# ============================================================================
# The dataset is already sorted by purchase timestamp (see 01_prepare_data.py).
#
# WHY TIME-BASED SPLIT BEATS RANDOM SPLIT HERE:
# ---------------------------------------------------------------
# We are predicting whether an order will be LATE.  In production the model
# will be used on *future* orders, not on a random sample from the past.
# A random split would leak future information into the training set
# (e.g. a seller's late behaviour that only emerged later would be used to
# train on early orders).  A time-based split that trains on the earliest
# 80% of orders and tests on the latest 20% mirrors the real deployment
# scenario and gives an honest estimate of out-of-time generalisation.
# Because the seller_hist_late_rate feature was already computed with an
# expanding window using only *strictly prior* orders, there is no
# look-ahead in the features; the time-based split now ensures the
# evaluation is also temporally honest.
#
# Note: order_id is generic, not timestamp-encoded, so we simply take
# the first 80% of rows as train and last 20% as test.

print("\n" + "=" * 65)
print("TIME-BASED SPLIT  (80% earliest → train, 20% latest → test)")
print("=" * 65)
split_idx = int(0.80 * len(df))
train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
test_df  = df.iloc[split_idx:].copy().reset_index(drop=True)

print(f"  Train: {len(train_df)} rows ({100*len(train_df)/len(df):.1f}%)")
print(f"  Test:  {len(test_df)} rows ({100*len(test_df)/len(df):.1f}%)")
print(f"  Train target:\n    late={train_df['target_late'].sum()}  on-time={(1-train_df['target_late']).sum()}")
print(f"  Test target:\n    late={test_df['target_late'].sum()}  on-time={(1-test_df['target_late']).sum()}")

# ============================================================================
# 3.  Feature / target separation
# ============================================================================
# Drop order_id (not a feature) and target_late (the target).
X_train = train_df.drop(columns=["order_id", "target_late"])
y_train = train_df["target_late"].values

X_test  = test_df.drop(columns=["order_id", "target_late"])
y_test  = test_df["target_late"].values

# Identify column types
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

print(f"\n  Numeric features ({len(numeric_cols)}): {numeric_cols}")
print(f"  Categorical features ({len(categorical_cols)}): {categorical_cols}")

# ============================================================================
# 4.  Build sklearn pipeline
# ============================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# ============================================================================
# 5.  Train models
# ============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -- Logistic Regression (baseline) ------------------------------------------
print("\n" + "=" * 65)
print("TRAINING: Logistic Regression (baseline)")
print("=" * 65)

lr_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)),
])
lr_pipe.fit(X_train, y_train)
print("  ✓ Fit complete")

# -- Random Forest (main model) ----------------------------------------------
print("\n" + "=" * 65)
print("TRAINING: Random Forest (main model)")
print("=" * 65)

rf_pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=5,
        random_state=42, n_jobs=-1, class_weight="balanced"
    )),
])
rf_pipe.fit(X_train, y_train)
print("  ✓ Fit complete")

# ============================================================================
# 6.  Evaluate both models
# ============================================================================
print("\n" + "=" * 65)
print("EVALUATION")
print("=" * 65)

results = {}

for name, pipe in [("LogisticRegression", lr_pipe), ("RandomForest", rf_pipe)]:
    print(f"\n  --- {name} ---")

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred)

    results[name] = {
        "roc_auc": round(roc_auc, 4),
        "f1":      round(f1, 4),
    }

    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  F1:      {f1:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['on_time', 'late'])}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{np.array2string(cm)}")

# Determine the better model based on ROC-AUC
if results["LogisticRegression"]["roc_auc"] >= results["RandomForest"]["roc_auc"]:
    best_name = "LogisticRegression"
    best_pipe = lr_pipe
else:
    best_name = "RandomForest"
    best_pipe = rf_pipe

print(f"\n  >>> Better model (by ROC-AUC): {best_name} "
      f"({results[best_name]['roc_auc']:.4f})")

# ============================================================================
# 7.  RandomForest feature importances (top 10)
# ============================================================================
print("\n" + "=" * 65)
print("RANDOM FOREST FEATURE IMPORTANCES  (top 10)")
print("=" * 65)

# Get feature names after transformation
cat_ohe = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"]
ohe_cat_names = cat_ohe.named_steps["ohe"].get_feature_names_out(categorical_cols)

all_feature_names = list(numeric_cols) + list(ohe_cat_names)

rf_clf = rf_pipe.named_steps["classifier"]
importances = rf_clf.feature_importances_

# Sort descending
top_indices = np.argsort(importances)[::-1][:10]
top_features = []
for rank, idx in enumerate(top_indices, 1):
    feat_name = all_feature_names[idx]
    imp_val = importances[idx]
    top_features.append({"rank": rank, "feature": feat_name, "importance": round(imp_val, 4)})
    print(f"  {rank:2d}. {feat_name:55s}  {imp_val:.4f}")

# ============================================================================
# 8.  Save the better model + metadata
# ============================================================================
import joblib

print(f"\n  Saving best model ({best_name}) to: {MODEL_PKL}")
joblib.dump(best_pipe, MODEL_PKL)

metadata = {
    "best_model": best_name,
    "metrics": results,
    "train_size": len(train_df),
    "test_size": len(test_df),
    "target_balance_train": {
        "late": int(train_df["target_late"].sum()),
        "on_time": int((1 - train_df["target_late"]).sum()),
    },
    "target_balance_test": {
        "late": int(test_df["target_late"].sum()),
        "on_time": int((1 - test_df["target_late"]).sum()),
    },
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "all_transformed_features": list(all_feature_names),
    "random_forest_top_10_features": top_features,
    "model_parameters": {
        "LogisticRegression": {
            "max_iter": 1000,
            "random_state": 42,
        },
        "RandomForest": {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "random_state": 42,
            "class_weight": "balanced",
        },
    },
}

with open(FEATURE_JSON, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  Saved metadata to: {FEATURE_JSON}")

print("\n" + "=" * 65)
print("DONE")
print("=" * 65)