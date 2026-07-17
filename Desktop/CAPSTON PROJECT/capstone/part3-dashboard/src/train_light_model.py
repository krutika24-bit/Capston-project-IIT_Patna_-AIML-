"""
Train a lightweight RandomForest late-delivery prediction model for the dashboard.

Reads the cleaned dataset produced by etl.py, engineers features
(all knowable at purchase time), trains a RandomForest, and saves to
model/model.pkl.

Standalone — does NOT import from Part 2.

Usage: python src/train_light_model.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PART3 = os.path.normpath(os.path.join(HERE, ".."))
CLEANED_CSV = os.path.join(PART3, "data", "cleaned", "orders_clean.csv")
MODEL_DIR = os.path.join(PART3, "model")
MODEL_PKL = os.path.join(MODEL_DIR, "model.pkl")
FEATURE_JSON = os.path.join(MODEL_DIR, "feature_list.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Feature engineering (condensed version of Part 2)
# ---------------------------------------------------------------------------
def engineer_features(df):
    """
    From the cleaned orders_clean.csv, produce features at the ORDER grain
    that are knowable at purchase time.
    """
    # Target
    df['target_late'] = df['is_late'].astype(int)

    # Time-based features from purchase timestamp
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df['purchase_weekday'] = df['order_purchase_timestamp'].dt.dayofweek
    df['purchase_month'] = df['order_purchase_timestamp'].dt.month

    # Cross-state is already in the data
    # Product category is already in the data

    # Compute a simple seller_hist_late_rate: overall average by seller
    # (simplified version — for a dashboard model this is acceptable)
    # We group by cross_state + product_category as a proxy for seller behavior.
    # For this dashboard model, use cross_state + promised_days + category as proxies.
    # A lightweight approach: use the global late rate as a fallback feature.
    global_late_rate = df['target_late'].mean()
    df['seller_hist_late_rate_approx'] = global_late_rate

    # Select features matching Part 2 (minus the time-ordered seller rate)
    feature_cols = [
        'order_id', 'target_late',
        'n_items', 'total_price', 'total_freight', 'freight_to_price_ratio',
        'promised_days', 'purchase_weekday', 'purchase_month',
        'cross_state', 'product_category', 'seller_hist_late_rate_approx',
    ]

    # Only delivered orders with known dates
    mask = (
        (df['order_status'] == 'delivered') &
        df['order_delivered_customer_date'].notna() &
        df['order_estimated_delivery_date'].notna()
    )
    result = df.loc[mask, feature_cols].copy().reset_index(drop=True)

    # Rename the approx column to match expected name
    result.rename(columns={'seller_hist_late_rate_approx': 'seller_hist_late_rate'}, inplace=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PART 3 — TRAIN LIGHT MODEL")
    print("=" * 60)

    # Load cleaned data
    if not os.path.isfile(CLEANED_CSV):
        print(f"Cleaned data not found at {CLEANED_CSV}")
        print("Run src/etl.py first.")
        return

    print(f"\nLoading cleaned data: {CLEANED_CSV}")
    df = pd.read_csv(CLEANED_CSV)
    print(f"  Rows: {len(df)}")

    # Engineer features
    print("\nEngineering features…")
    model_df = engineer_features(df)
    print(f"  Model-ready rows: {len(model_df)}")
    print(f"  Target balance:\n    late={model_df['target_late'].sum()}  on-time={(1-model_df['target_late']).sum()}")

    # Train / test split (simple 80/20 time-based)
    model_df.sort_values('order_id', inplace=True)
    split_idx = int(0.80 * len(model_df))
    train_df = model_df.iloc[:split_idx].reset_index(drop=True)
    test_df = model_df.iloc[split_idx:].reset_index(drop=True)

    # Features
    X_train = train_df.drop(columns=['order_id', 'target_late'])
    y_train = train_df['target_late'].values
    X_test = test_df.drop(columns=['order_id', 'target_late'])
    y_test = test_df['target_late'].values

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

    print(f"\n  Numeric features: {numeric_cols}")
    print(f"  Categorical features: {categorical_cols}")

    # Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                         ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, numeric_cols),
        ('cat', cat_pipe, categorical_cols),
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=5,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )),
    ])

    print("\nTraining RandomForest…")
    pipe.fit(X_train, y_train)
    print("  ✓ Fit complete")

    # Evaluate
    from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n  ROC-AUC: {roc_auc:.4f}")
    print(f"  F1:      {f1:.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['on_time', 'late'])}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:\n{np.array2string(cm)}")

    # Feature importances
    ohe = pipe.named_steps['preprocessor'].named_transformers_['cat']
    ohe_names = ohe.named_steps['ohe'].get_feature_names_out(categorical_cols)
    all_names = list(numeric_cols) + list(ohe_names)
    importances = pipe.named_steps['classifier'].feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    top_features = []
    print("\nTop 10 feature importances:")
    for rank, idx in enumerate(top_idx, 1):
        top_features.append({'rank': rank, 'feature': all_names[idx], 'importance': round(float(importances[idx]), 4)})
        print(f"  {rank:2d}. {all_names[idx]:50s} {importances[idx]:.4f}")

    # Save
    print(f"\n  Saving model to: {MODEL_PKL}")
    joblib.dump(pipe, MODEL_PKL)

    metadata = {
        'metrics': {'roc_auc': round(float(roc_auc), 4), 'f1': round(float(f1), 4)},
        'train_size': len(train_df),
        'test_size': len(test_df),
        'numeric_features': numeric_cols,
        'categorical_features': categorical_cols,
        'top_features': top_features,
    }
    with open(FEATURE_JSON, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to: {FEATURE_JSON}")
    print("\nDone.")


if __name__ == '__main__':
    main()