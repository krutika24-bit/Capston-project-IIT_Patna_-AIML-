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

This dashboard helps merchandising and operations managers make data-driven decisions about delivery performance. It uses historical order data to show where delays happen, why they happen, and how likely a new order is to be late.

The dashboard is organized into three tabs, each answering a specific business question.

---

## Tab 1: Overview — “Where are we strong and where are we weak?”

**What you see:**
- Three headline numbers at the top: total sales revenue, average delivery time, and average customer review score.
- A bar chart showing revenue by customer state.
- A histogram showing how many orders took 0–5 days, 5–10 days, etc., color-coded by whether they were late.

**Business decision it supports:**
Use this view to decide which states or regions deserve more logistics investment. If a state generates high revenue but also has slow deliveries or low review scores, that’s a signal to improve carrier selection, add a local warehouse, or adjust delivery promises.

---

## Tab 2: Diagnostic — “What is causing delays, and what do they cost us?”

**What you see:**
- Two horizontal bar charts: one ranking states by late-delivery rate, and one ranking product categories by late-delivery rate.
- A scatter plot where each dot is an order, showing the relationship between delivery speed (x-axis) and customer review score (y-axis). Late orders are highlighted in red.

**Business decision it supports:**
Use this tab to prioritize operational fixes. If one product category is consistently late, you might negotiate better shipping terms or change packaging. The scatter plot answers the question “Do late deliveries actually hurt customer satisfaction?” — if the red dots cluster at lower review scores, the answer is yes, and fixing delays should be a high priority.

---

## Tab 3: Predict — “Is this specific order at risk of being late?”

**What you see:**
- A simple form where you enter order details: number of items, price, freight cost, promised delivery window, purchase day/month, whether the shipment crosses state lines, and product category.
- A prediction result showing the probability that the order will be late, plus a bar chart of the top factors contributing to that risk.

**Business decision it supports:**
Before promising a delivery date to a customer, operations staff can estimate delay risk and take proactive action — for example, flagging high-risk orders for priority handling, informing the customer of a possible delay upfront, or adjusting the promised delivery window to set realistic expectations.

---

## Model Details

- **Algorithm:** Random Forest
- **Goal:** Predict whether an order will be delivered after the estimated date
- **Key inputs:** order value, shipping cost, delivery promise, product category, shipping distance (cross-state flag), and the seller’s historical late rate
- **Performance:** ROC-AUC displayed in the Predict tab

---

## Tech Stack

- **Streamlit** — interactive dashboard
- **Plotly** — interactive charts
- **scikit-learn / joblib** — model inference
- **pandas / numpy** — data processing