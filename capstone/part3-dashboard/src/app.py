"""
Streamlit dashboard for Olist E-Commerce delivery analytics.

3 tabs:
  1. Overview   — KPIs with date/state filters + revenue by state chart
  2. Diagnostic — late rate by state/category + review vs delivery scatter
  3. Predict    — form that calls the saved model and shows risk + contributions

Usage: streamlit run src/app.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Olist Delivery Analytics",
    page_icon="📦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PART3 = os.path.normpath(os.path.join(HERE, ".."))
CLEANED_CSV = os.path.join(PART3, "data", "cleaned", "orders_clean.csv")
MODEL_PKL = os.path.join(PART3, "model", "model.pkl")
FEATURE_JSON = os.path.join(PART3, "model", "feature_list.json")


# ---------------------------------------------------------------------------
# Cached data + model loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(CLEANED_CSV)
    for col in ['order_purchase_timestamp', 'order_delivered_customer_date',
                'order_estimated_delivery_date', 'order_approved_at']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['delivery_days'] = pd.to_numeric(df['delivery_days'], errors='coerce')
    df['avg_review_score'] = df['avg_review_score'].fillna(0)
    # Only delivered for analysis
    delivered = df[df['order_status'] == 'delivered'].copy()
    return df, delivered


@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PKL):
        return None, None
    model = joblib.load(MODEL_PKL)
    with open(FEATURE_JSON) as f:
        meta = json.load(f)
    return model, meta


@st.cache_data
def get_category_list():
    """Sorted unique product categories from the data."""
    _, d = load_data()
    return sorted(d['product_category'].dropna().unique())


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df_all, df = load_data()
model, model_meta = load_model()
categories = get_category_list()

# Global late rate for the default seller_hist_late_rate
GLOBAL_LATE_RATE = float(df['is_late'].mean())

# ===========================================================================
# TAB 1 — OVERVIEW
# ===========================================================================
def tab_overview():
    st.title("📦 Olist Delivery Analytics — Overview")

    # ---- Filters ----
    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        min_date = df['order_purchase_timestamp'].min().date()
        max_date = df['order_purchase_timestamp'].max().date()
        date_range = st.date_input("Date range", [min_date, max_date])
    with col_f2:
        states = sorted(df['customer_state'].dropna().unique())
        selected_states = st.multiselect("State(s)", states, default=states[:5])
    with col_f3:
        pass  # placeholder

    # Apply filters
    mask = df['customer_state'].isin(selected_states)
    if len(date_range) == 2:
        mask &= (df['order_purchase_timestamp'].dt.date >= date_range[0]) & \
                (df['order_purchase_timestamp'].dt.date <= date_range[1])
    filtered = df[mask]

    if filtered.empty:
        st.warning("No data matches the selected filters.")
        return

    # ---- KPIs ----
    total_revenue = filtered['total_price'].sum()
    avg_delivery = filtered['delivery_days'].mean()
    avg_review = filtered['avg_review_score'].mean()

    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    with k2:
        st.metric("Avg Delivery Days", f"{avg_delivery:.1f}")
    with k3:
        st.metric("Avg Review Score", f"{avg_review:.2f} / 5.0")

    st.markdown("---")

    # ---- Revenue by state ----
    st.subheader("💰 Revenue by Customer State")
    state_agg = filtered.groupby('customer_state').agg(
        revenue=('total_price', 'sum'),
        orders=('order_id', 'nunique'),
    ).reset_index().sort_values('revenue', ascending=False)

    fig1 = px.bar(
        state_agg,
        x='customer_state', y='revenue',
        color='orders',
        color_continuous_scale='Blues',
        labels={'customer_state': 'State', 'revenue': 'Revenue ($)'},
        text_auto='.0f',
        title='Revenue per State (color = order count)',
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---- Delivery time distribution ----
    st.subheader("📊 Delivery Time Distribution")
    fig2 = px.histogram(
        filtered,
        x='delivery_days',
        color='is_late',
        nbins=30,
        barmode='overlay',
        opacity=0.6,
        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
        labels={'delivery_days': 'Days', 'count': 'Orders', 'is_late': 'Late'},
    )
    st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# TAB 2 — DIAGNOSTIC
# ===========================================================================
def tab_diagnostic():
    st.title("🔍 Diagnostic — Late Delivery Drivers")

    col1, col2 = st.columns(2)

    # ---- Late rate by state ----
    with col1:
        st.subheader("🗺️ Late Rate by State")
        state_late = df.groupby('customer_state').agg(
            late_rate=('is_late', 'mean'),
            orders=('order_id', 'count'),
        ).reset_index()
        state_late = state_late[state_late['orders'] >= 5].sort_values('late_rate', ascending=False)

        fig3 = px.bar(
            state_late.head(15),
            x='late_rate', y='customer_state',
            color='late_rate',
            color_continuous_scale='RdYlGn_r',
            text_auto='.0%',
            labels={'late_rate': 'Late Rate', 'customer_state': ''},
            orientation='h',
            title='Top 15 States by Late Rate',
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)

    # ---- Late rate by category ----
    with col2:
        st.subheader("🏷️ Late Rate by Product Category")
        cat_late = df.groupby('product_category').agg(
            late_rate=('is_late', 'mean'),
            orders=('order_id', 'count'),
        ).reset_index()
        cat_late = cat_late[cat_late['orders'] >= 10].sort_values('late_rate', ascending=False)

        fig4 = px.bar(
            cat_late.head(15),
            x='late_rate', y='product_category',
            color='late_rate',
            color_continuous_scale='RdYlGn_r',
            text_auto='.0%',
            labels={'late_rate': 'Late Rate', 'product_category': ''},
            orientation='h',
            title='Top 15 Categories by Late Rate (min 10 orders)',
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ---- Scatter: Review Score vs Delivery Days ----
    st.subheader("⭐ Review Score vs Delivery Days")
    reviewed = df[df['avg_review_score'] > 0].copy()

    fig5 = px.scatter(
        reviewed.sample(min(2000, len(reviewed)), random_state=42),
        x='delivery_days',
        y='avg_review_score',
        color='is_late',
        color_discrete_map={0: '#00CC96', 1: '#EF553B'},
        opacity=0.5,
        labels={'delivery_days': 'Delivery Days', 'avg_review_score': 'Review Score', 'is_late': 'Late'},
        title='Each dot = one delivered order (sampled)',
    )
    st.plotly_chart(fig5, use_container_width=True)


# ===========================================================================
# TAB 3 — PREDICT
# ===========================================================================
def tab_predict():
    st.title("🎯 Predict — Will This Order Be Late?")

    if model is None:
        st.error("Model not found. Run `src/train_light_model.py` first.")
        return

    st.markdown("Enter order details below to get a late-delivery risk assessment.")

    col1, col2, col3 = st.columns(3)

    with col1:
        n_items = st.number_input("Number of items", min_value=1, max_value=50, value=2)
        total_price = st.number_input("Total price ($)", min_value=1.0, value=150.0, step=10.0)
        total_freight = st.number_input("Freight cost ($)", min_value=0.0, value=20.0, step=5.0)

    with col2:
        promised_days = st.number_input("Promised delivery (days)", min_value=1, max_value=60, value=15)
        weekday = st.selectbox("Purchase weekday", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], index=2)
        month = st.selectbox("Purchase month", list(range(1, 13)), format_func=lambda m: datetime(2000, m, 1).strftime('%B'), index=4)

    with col3:
        cross_state = st.selectbox("Cross-state shipment?", ["No", "Yes"], index=0)
        payment_installments = st.number_input("Payment installments", min_value=1, max_value=24, value=4)
        product_category = st.selectbox("Product category", categories, index=0)

    predict_btn = st.button("🚀 Predict", type="primary")

    if predict_btn:
        weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        freight_ratio = round(total_freight / total_price, 4) if total_price > 0 else 0.0

        input_dict = {
            "n_items": n_items,
            "total_price": total_price,
            "total_freight": total_freight,
            "freight_to_price_ratio": freight_ratio,
            "promised_days": promised_days,
            "purchase_weekday": weekday_map[weekday],
            "purchase_month": month,
            "cross_state": 1 if cross_state == "Yes" else 0,
            "product_category": product_category,
            "seller_hist_late_rate": GLOBAL_LATE_RATE,
        }

        input_df = pd.DataFrame([input_dict])
        proba = float(model.predict_proba(input_df)[0, 1])
        pred = int(proba >= 0.5)

        # ---- Show result ----
        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown("### Prediction Result")
            if pred == 1:
                st.error(f"🔴 **LATE**  — {proba:.1%} probability")
            else:
                st.success(f"🟢 **ON TIME**  — {1-proba:.1%} probability")

            st.progress(float(proba), text="Late probability")

        with res_col2:
            st.markdown("### Top Contributing Features")
            st.caption("Global feature importances from the Random Forest model")

            if model_meta and 'top_features' in model_meta:
                tf = pd.DataFrame(model_meta['top_features'])
                tf['feature'] = tf['feature'].str.slice(0, 45)
                tf['color'] = tf['importance'].apply(lambda x: '#EF553B' if x > 0.03 else '#636EFA')

                fig6 = px.bar(
                    tf,
                    y='feature',
                    x='importance',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Viridis',
                    labels={'feature': '', 'importance': 'Importance'},
                    title=f'Model ROC-AUC: {model_meta["metrics"]["roc_auc"]:.3f}',
                )
                fig6.update_layout(height=350, showlegend=False)
                st.plotly_chart(fig6, use_container_width=True)

        # ---- Input summary ----
        with st.expander("📋 Input values", expanded=False):
            summary = pd.DataFrame([input_dict])
            summary = summary.drop(columns=['seller_hist_late_rate'])
            st.dataframe(summary, use_container_width=True)


# ===========================================================================
# TABS
# ===========================================================================
tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Diagnostic", "🎯 Predict"])

with tab1:
    tab_overview()

with tab2:
    tab_diagnostic()

with tab3:
    tab_predict()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"📦 Olist Delivery Analytics • "
    f"{len(df):,} delivered orders • "
    f"Model ROC-AUC: {model_meta['metrics']['roc_auc']:.3f} • "
    f"Built with Streamlit & Plotly"
)