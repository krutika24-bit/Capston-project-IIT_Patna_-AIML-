# Olist Delivery Analytics Dashboard

## 🌐 Live Dashboard

**Streamlit App:** https://capston-project-iit-patna-aiml.streamlit.app

*Deployed on share.streamlit.io — no local setup required.*

---

## Run Locally

If you prefer to run the dashboard on your own machine:

```bash
# 1. Clone the repository
git clone https://github.com/krutika24-bit/Capston-project-IIT_Patna_-AIML-.git
cd Capston-project-IIT_Patna_-AIML-/capstone/part3-dashboard

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r ../../requirements.txt

# 4. Ensure the following data files exist:
#    - ../data/cleaned/orders_clean.csv
#    - model/model.pkl
#    - model/feature_list.json
#    Run the data pipeline scripts first if these are missing (see repo root README).

# 5. Launch the dashboard
streamlit run src/app.py
```

---

## About This Dashboard

This dashboard helps merchandising and operations managers make data-driven decisions about delivery performance. It uses historical order data to show where delays happen, why they happen, and how they impact customer satisfaction.

The dashboard is organized into three tabs, each answering a specific business question.

---

## Tab 1: Overview — "Where are we strong and where are we weak?"

**What you see:**
- Three headline numbers at the top: total sales revenue, average delivery time, and average customer review score.
- A bar chart showing revenue by customer state.
- A histogram showing how many orders took 0–5 days, 5–10 days, etc., color-coded by whether they were late.

**Business decision it supports:**
Use this view to decide which states or regions deserve more logistics investment. If a state generates high revenue but also has slow deliveries or low review scores, that's a signal to improve logistics in that region.

---

## Tab 2: Diagnostic — "What is causing delays, and what do they cost us?"

**What you see:**
- Two horizontal bar charts: one ranking states by late-delivery rate, and one ranking product categories by late-delivery rate.
- A scatter plot where each dot is an order, showing the relationship between delivery speed (x-axis) and customer review score (y-axis). Late orders are highlighted in red.

**Business decision it supports:**
Use this tab to prioritize operational fixes. If one product category is consistently late, you might negotiate better shipping terms or change packaging. The scatter plot answers the question "What's the cost of a late delivery in terms of lost customer satisfaction?"

---

## Tab 3: Predict — "Is this specific order at risk of being late?"

**What you see:**
- A simple form where you enter order details: number of items, price, freight cost, promised delivery window, purchase day/month, whether the shipment crosses state lines, and product category.
- A prediction result showing the probability that the order will be late, plus a bar chart of the top factors contributing to that risk.

**Business decision it supports:**
Before promising a delivery date to a customer, operations staff can estimate delay risk and take proactive action — for example, flagging high-risk orders for priority handling, informing the customer of extended timelines, or re-routing the shipment.

---

## Model Details

- **Algorithm:** Logistic Regression ⭐ (best performer, selected over Random Forest)
- **Goal:** Predict whether an order will be delivered after the estimated date
- **Key inputs:** order value, shipping cost, delivery promise, product category, shipping distance (cross-state flag), and the seller's historical late rate
- **Performance:** ROC-AUC = 0.7273 (displayed in the Predict tab)

### Model Selection Rationale

Both Logistic Regression and Random Forest were trained and evaluated:

| Model | ROC-AUC | F1 | Reason Selected |
|-------|---------|-----|---|
| **Logistic Regression** ⭐ | **0.7273** | **0.6774** | Simpler, more interpretable, higher ROC-AUC (0.0058 advantage) |
| Random Forest | 0.7215 | 0.6760 | Comparable performance but less transparent |

Logistic Regression was chosen because:
1. **Marginally better performance** — 0.7273 vs 0.7215 ROC-AUC
2. **Model interpretability** — Coefficients can be explained to stakeholders
3. **Fairness & debugging** — Linear decision boundary is easier to audit
4. **Deployment** — Smaller model, faster inference, fewer dependencies

---

## Tech Stack

- **Streamlit** — interactive dashboard
- **Plotly** — interactive charts
- **scikit-learn / joblib** — model inference (Logistic Regression)
- **pandas / numpy** — data processing
