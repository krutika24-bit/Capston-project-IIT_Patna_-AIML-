# Olist Delivery Analytics Dashboard

## 🌐 Live Dashboard

**Streamlit App:** [https://share.streamlit.io/](https://share.streamlit.io/)  
*Replace with the actual URL after deployment*

---

## Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/krutika24-bit/Capston-project-IIT_Patna_-AIML-.git
cd Capston-project-IIT_Patna_-AIML-/capstone/part3-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure the following files exist (from the data pipeline):
#    - ../data/cleaned/orders_clean.csv
#    - model/model.pkl
#    - model/feature_list.json

# 4. Run the dashboard
streamlit run src/app.py
```

---

## Dashboard Overview for Business Users

This dashboard helps merchandising and operations managers understand delivery performance and predict late orders. It is organized into three tabs, each supporting a specific business decision.

---

### Tab 1: Overview

**What it shows:**
- High-level KPIs: total revenue, average delivery time, and average customer review score
- Revenue breakdown by customer state
- Distribution of delivery times, highlighting which orders were late

**Decision it supports:**
"Where should we focus our logistics investments?"

Use this tab to identify high-revenue states and see whether delivery performance varies across regions. If certain states show lower average review scores or higher late rates, they may need better carrier partnerships or warehouse placement.

---

### Tab 2: Diagnostic

**What it shows:**
- Late delivery rate by state and by product category
- Scatter plot linking delivery speed to customer satisfaction

**Decision it supports:**
"Why are orders late, and what is the business impact?"

This tab pinpoints the states and product categories with the worst on-time performance. The scatter plot quantifies the satisfaction penalty of late deliveries, helping managers prioritize fixes that directly improve customer reviews.

---

### Tab 3: Predict

**What it shows:**
- A simple input form to score a new order for late-delivery risk
- Probability of lateness and the top factors driving that risk
- Model performance metrics (ROC-AUC)

**Decision it "What is the risk of this specific order being late?"

Before promising a delivery date to a customer, operations staff can estimate whether that order is likely to arrive late based on its characteristics (price, freight cost, category, shipping distance, etc.). This supports proactive communication and exception handling.

---

## Model Information

- **Algorithm:** Random Forest
- **Target:** binary indicator of whether the order was delivered after the estimated date
- **Performance:** ROC-AUC score displayed in the Predict tab
- **Features used:** number of items, price, freight cost, freight-to-price ratio, promised delivery window, purchase weekday/month, cross-state flag, product category, and historical seller late rate

---

## Repository Setup

1. The dashboard reads data from the shared data pipeline (`../data/cleaned/orders_clean.csv`) and model artifacts (`model/model.pkl`, `model/feature_list.json`).
2. Run the data preparation and model training scripts first if those files are missing. See the parent README for details.

---

## Tech Stack

- **Streamlit** — interactive web app framework
- **Plotly** — charts and visualizations
- **scikit-learn / joblib** — model inference
- **pandas / numpy** — data handling