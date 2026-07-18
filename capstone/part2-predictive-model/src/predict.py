"""
Score a single order for late-delivery probability.

Exposes score_order(order: dict) -> dict that loads the saved model
and feature metadata, transforms the input, and returns the prediction.

Usage (CLI):
    python src/predict.py

Usage (import):
    from predict import score_order
    result = score_order({
        "n_items": 2,
        "total_price": 150.00,
        ...
    })
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PART2 = os.path.normpath(os.path.join(HERE, ".."))
MODEL_PKL = os.path.join(PART2, "model", "model.pkl")
FEATURE_JSON = os.path.join(PART2, "model", "feature_list.json")


def _load_artifacts():
    """Lazy-load model and metadata, caching after first call."""
    if not hasattr(_load_artifacts, "_model"):
        if not os.path.isfile(MODEL_PKL):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PKL}. Run src/02_train_eval.py first."
            )
        if not os.path.isfile(FEATURE_JSON):
            raise FileNotFoundError(
                f"Feature metadata not found at {FEATURE_JSON}. "
                "Run src/02_train_eval.py first."
            )
        _load_artifacts._model = joblib.load(MODEL_PKL)
        with open(FEATURE_JSON) as f:
            _load_artifacts._meta = json.load(f)
    return _load_artifacts._model, _load_artifacts._meta


def score_order(order: dict) -> dict:
    """
    Score a single order for late-delivery probability.

    Parameters
    ----------
    order : dict
        Must contain all features the model expects:
          numeric: n_items, total_price, total_freight, freight_to_price_ratio,
                   payment_installments, promised_days, purchase_weekday,
                   purchase_month, cross_state, seller_hist_late_rate
          categorical: payment_type, product_category

    Returns
    -------
    dict with keys:
        late_probability : float  (0-1)
        predicted_late   : int    (1 if late_probability >= 0.5 else 0)
    """
    model, meta = _load_artifacts()

    # Build a single-row DataFrame with the exact columns the pipeline expects
    num_cols = meta["numeric_features"]
    cat_cols = meta["categorical_features"]
    all_cols = num_cols + cat_cols

    # Validate input keys
    missing = [c for c in all_cols if c not in order]
    if missing:
        raise KeyError(
            f"Missing required feature(s) in order dict: {missing}. "
            f"Expected: {all_cols}"
        )

    row = {col: order[col] for col in all_cols}
    df = pd.DataFrame([row])

    # Predict
    proba = model.predict_proba(df)[0, 1]   # probability of class 1 (late)
    pred  = int(proba >= 0.5)

    return {
        "late_probability": round(float(proba), 4),
        "predicted_late": pred,
    }


# ============================================================================
# CLI example
# ============================================================================
if __name__ == "__main__":
    # A worked example order — values taken from a real row in the dataset
    example_order = {
        "n_items": 3,
        "total_price": 275.80,
        "total_freight": 42.15,
        "freight_to_price_ratio": 0.1528,
        "payment_installments": 4,
        "payment_type": "credit_card",
        "promised_days": 18,
        "purchase_weekday": 2,       # Wednesday
        "purchase_month": 5,         # May
        "cross_state": 1,
        "product_category": "beleza_saude",
        "seller_hist_late_rate": 0.55,
    }

    print("=" * 60)
    print("SCORE ORDER — Worked Example")
    print("=" * 60)
    print("\nInput order:")
    for k, v in example_order.items():
        print(f"  {k:30s} = {v}")

    try:
        result = score_order(example_order)
        print(f"\nResult:")
        print(f"  late_probability : {result['late_probability']}")
        print(f"  predicted_late   : {result['predicted_late']}  "
              f"({'LATE' if result['predicted_late'] else 'ON TIME'})")
    except Exception as e:
        print(f"\nERROR: {e}")