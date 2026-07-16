# Part 2 — Predictive Model: Late-Delivery Prediction

## Business Framing

This Part answers the **PREDICTIVE** question ("what will happen"):

> **Given an order at the moment it is placed, can we predict whether it will be delivered late?**

In Part 1 we diagnosed that late deliveries cause a **1.54-point drop** in customer review scores (from 4.20 to 2.66). If we can identify high-risk orders at purchase time, the business can intervene proactively — for example by flagging the order for expedited handling, adjusting the promised delivery date, or alerting the seller.

---

## Target Definition

| Column | Definition |
|---|---|
| `target_late` | `1` if `order_delivered_customer_date > order_estimated_delivery_date`, else `0` |

Only **delivered** orders with both actual and estimated delivery dates are included (4,514 out of 5,000 total orders). The target is reasonably balanced: 52% late, 48% on-time.

---

## Leakage Precautions

Time-series prediction is notoriously vulnerable to data leakage — using information that would not be available at prediction time. The following precautions were taken:

### 1. `seller_hist_late_rate` — Expanding window, time-ordered

This is the most leakage-prone feature. A naive implementation might compute each seller's overall late rate across all orders, which would use *future* orders to predict *past* ones. Instead:

- Orders are sorted by `order_purchase_timestamp` (ascending).
- For each order, the seller's late rate is computed using **only orders strictly before** the current one.
- If a seller has fewer than 3 prior orders, the feature falls back to the **global running average** (over all sellers, also computed from strictly prior orders).
- The current order's outcome is only added to the seller's history **after** computing the feature for that order.

This means `seller_hist_late_rate` for order N is exactly what you would know about that seller at the moment order N is placed.

### 2. Time-based train/test split

A random split would train on some future orders and test on some past ones, giving an unrealistically optimistic performance estimate. Instead:

- **80% earliest orders** (by purchase timestamp) → training set
- **20% latest orders** → test set

This mirrors the real deployment scenario: train on historical data, predict on future orders. The model never sees test-set temporal patterns during training.

### 3. Features knowable at purchase time

All features are derived from information available when the order is placed:

| Feature | Available at purchase? |
|---|---|
| `n_items`, `total_price`, `total_freight` | Yes — from the cart contents |
| `freight_to_price_ratio` | Yes — computed from cart |
| `payment_installments`, `payment_type` | Yes — chosen by customer |
| `promised_days` | Yes — shown to customer |
| `purchase_weekday`, `purchase_month` | Yes — current date |
| `cross_state` | Yes — known from customer/seller addresses |
| `product_category` | Yes — from the product catalog |
| `seller_hist_late_rate` | Yes — computed from prior orders only |

No future information (review scores, actual delivery dates, etc.) is used as a feature.

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare the modeling dataset
python src/01_prepare_data.py

# 3. Train and evaluate models
python src/02_train_eval.py

# 4. Score a single order (worked example)
python src/predict.py
```

The scripts are **standalone** — they do not import anything from Part 1. If no DuckDB database is found, `01_prepare_data.py` will generate a synthetic Olist dataset automatically.

---

## Model Comparison

| Model | ROC-AUC | F1 | Precision (late) | Recall (late) |
|---|---|---|---|---|
| **Logistic Regression** | **0.7273** | **0.6774** | 0.66 | 0.70 |
| Random Forest | 0.7215 | 0.6760 | 0.66 | 0.69 |

**Better model saved:** `LogisticRegression` (by 0.0058 ROC-AUC)

Both models perform similarly, with the simpler logistic regression edging ahead. This suggests the decision boundary is approximately linear in the feature space.

### Why ROC-AUC and F1 over Accuracy?

- **Accuracy** is misleading when classes are imbalanced — a model that always predicts "late" would achieve 52% accuracy here, which sounds reasonable but is useless.
- **ROC-AUC** measures the model's ability to rank orders by risk across all thresholds. A score of 0.73 means there is a 73% chance the model will assign a higher risk score to a randomly chosen late order than to a randomly chosen on-time order.
- **F1** balances precision and recall. For this use case, **recall** (catching late orders) is more important than precision (avoiding false alarms), because the cost of a false positive (unnecessary expediting) is lower than the cost of a false negative (an unhappy customer).

---

## Feature Importances (Random Forest)

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | `promised_days` | 0.4978 | **The promised delivery window dominates.** Orders with longer estimates are more likely to be late — consistent with Part 1's finding that delivery times vary by ~6 days across categories. |
| 2 | `freight_to_price_ratio` | 0.0684 | Higher freight relative to price may indicate remote locations or bulky items, both risk factors for delay. |
| 3 | `total_freight` | 0.0656 | Absolute shipping cost correlates with distance and item size. |
| 4 | `seller_hist_late_rate` | 0.0627 | **A seller's past punctuality is a strong signal.** This validates the expanding-window feature engineering approach. |
| 5 | `total_price` | 0.0613 | Higher-value orders may receive different handling, or may correlate with different product categories. |
| 6 | `purchase_month` | 0.0337 | Seasonal effects (holiday rushes, weather) impact logistics. |
| 7 | `payment_installments` | 0.0328 | More installments may correlate with higher-value or cross-border purchases. |
| 8 | `purchase_weekday` | 0.0282 | Orders placed on different days may have different processing delays. |
| 9 | `n_items` | 0.0171 | Multi-item orders require consolidation, adding complexity. |
| 10 | `payment_type_debit_card` | 0.0083 | Payment method may correlate with customer segment or country. |

### Connection to Part 1

Part 1's diagnostic analysis found that **late deliveries cause a 1.54-point drop in review scores** and that **~52% of all delivered orders arrive late**. The top feature here — `promised_days` — directly ties back to that finding: the delivery estimate the seller sets is the single strongest predictor of whether the order will actually arrive on time. This suggests that improving estimate accuracy (rather than just speed) could be a high-impact intervention.

---

## Files

```
capstone/part2-predictive-model/
├── requirements.txt          # duckdb, pandas, numpy, scikit-learn, joblib
├── README.md                 # This file
├── src/
│   ├── 01_prepare_data.py    # Builds model_dataset.csv at ORDER grain
│   ├── 02_train_eval.py      # Trains/evaluates/saves models
│   └── predict.py            # score_order() function + CLI example
├── data/
│   ├── model_dataset.csv     # 4,514 rows x 14 columns
│   └── olist.duckdb          # (auto-generated if missing)
└── model/
    ├── model.pkl             # Serialized best model (LogisticRegression)
    └── feature_list.json     # Full metadata
```

---

## AI Assistance Disclosure

This pipeline was developed with the assistance of **Claude Code (Anthropic)**, an AI coding assistant. Claude helped generate boilerplate code, structure the sklearn pipeline, format the evaluation output, and draft this README. All logic, design decisions, and analytical interpretations were reviewed and validated by the author.