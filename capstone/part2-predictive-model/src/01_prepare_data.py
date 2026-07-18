"""
Prepare a modeling dataset at the ORDER grain for late-delivery prediction.

Target: target_late = 1 if order_delivered_customer_date >
        order_estimated_delivery_date, else 0.

Features (all knowable at order-purchase time):
  - n_items, total_price, total_freight, freight_to_price_ratio
  - payment_installments, payment_type
  - promised_days
  - purchase_weekday, purchase_month
  - cross_state
  - product_category (most common in the order)
  - seller_hist_late_rate: expanding time-ordered average with fallback

Output: data/model_dataset.csv
         Prints target balance to stdout.

Usage:  python src/01_prepare_data.py

Standalone — does NOT import anything from part1-data-engineering.
"""

import os
import sys
import math
import duckdb
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths  (relative to this script's location)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # repo root
PART2 = os.path.normpath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(PART2, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# Where to find / create the DuckDB
DB_CANDIDATES = [
    os.path.join(PART2, "data", "olist.duckdb"),          # local to part2
    os.path.join(REPO, "data", "olist.duckdb"),           # repo root (part1 generated)
]
OUTPUT_CSV = os.path.join(DATA_DIR, "model_dataset.csv")


# ---------------------------------------------------------------------------
# 1.  Locate or build the Olist DuckDB
# ---------------------------------------------------------------------------
def find_or_build_db():
    for p in DB_CANDIDATES:
        if os.path.isfile(p):
            print(f"[01_prepare_data] Found database: {p}")
            return p

    # No existing DB found — build one from raw CSVs or synthetic generator
    print("[01_prepare_data] No Olist DuckDB found. Building synthetic dataset…")

    import random
    from datetime import datetime, timedelta

    random.seed(42)

    STATES = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'ES',
              'PE', 'CE', 'PA', 'MT', 'MS', 'MA', 'PB', 'AL', 'RN', 'PI',
              'SE', 'RO', 'TO', 'AC', 'AM', 'RR', 'AP']

    CITIES_BY_STATE = {
        s: [f"city_{s}_{i}" for i in range(1, 6)] for s in STATES
    }

    PRODUCT_CATEGORIES = [
        'beleza_saude', 'moveis_decoracao', 'informatica_acessorios',
        'cama_mesa_banho', 'esporte_lazer', 'telefonia',
        'relogios_presentes', 'automotivo', 'brinquedos',
        'ferramentas_jardim', 'eletrodomesticos', 'livros_interesse',
        'eletronicos', 'alimentos_bebidas', 'bebes',
        'papelaria', 'moda_roupa', 'casa_conforto',
        'perfumaria', 'utilidades_domesticas', 'consoles_games',
    ]

    PAYMENT_TYPES = ['credit_card', 'boleto', 'voucher', 'debit_card']

    db_path = os.path.join(PART2, "data", "olist.duckdb")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = duckdb.connect(str(db_path))

    n_customers = 2000
    customers = []
    for i in range(1, n_customers + 1):
        st = random.choice(STATES)
        customers.append({
            'customer_id': f'C{i:05d}',
            'customer_unique_id': f'U{random.randint(10000,99999):05d}',
            'customer_zip_code_prefix': random.randint(10000, 99999),
            'customer_city': random.choice(CITIES_BY_STATE[st]),
            'customer_state': st,
        })
    df_c = pd.DataFrame(customers)
    conn.execute("CREATE OR REPLACE TABLE olist_customers AS SELECT * FROM df_c")

    n_sellers = 300
    sellers = []
    for i in range(1, n_sellers + 1):
        st = random.choice(STATES)
        sellers.append({
            'seller_id': f'S{i:05d}',
            'seller_zip_code_prefix': random.randint(10000, 99999),
            'seller_city': random.choice(CITIES_BY_STATE[st]),
            'seller_state': st,
        })
    df_s = pd.DataFrame(sellers)
    conn.execute("CREATE OR REPLACE TABLE olist_sellers AS SELECT * FROM df_s")

    n_products = 500
    products = []
    for i in range(1, n_products + 1):
        products.append({
            'product_id': f'P{i:05d}',
            'product_category_name': random.choice(PRODUCT_CATEGORIES),
            'product_name_length': random.randint(20, 80),
            'product_description_length': random.randint(100, 1000),
            'product_photos_qty': random.randint(1, 8),
            'product_weight_g': random.randint(100, 5000),
            'product_length_cm': random.randint(10, 100),
            'product_height_cm': random.randint(5, 50),
            'product_width_cm': random.randint(10, 60),
        })
    df_p = pd.DataFrame(products)
    conn.execute("CREATE OR REPLACE TABLE olist_products AS SELECT * FROM df_p")

    def rand_date(start, end):
        delta = end - start
        return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

    n_orders = 5000
    start_date = datetime(2016, 9, 1)
    end_date = datetime(2018, 10, 31)
    orders = []

    for i in range(1, n_orders + 1):
        cust_id = f'C{random.randint(1, n_customers):05d}'
        p_ts = rand_date(start_date, end_date)
        a_ts = p_ts + timedelta(hours=random.randint(1, 48))
        roll = random.random()
        if roll < 0.90:
            status = 'delivered'
            d_days = random.randint(3, 30)
            d_ts = p_ts + timedelta(days=d_days)
            e_ts = p_ts + timedelta(days=random.randint(5, 25))
        elif roll < 0.95:
            status = 'shipped'
            d_ts = None
            e_ts = p_ts + timedelta(days=random.randint(5, 25))
        elif roll < 0.98:
            status = 'canceled'
            d_ts = None
            e_ts = None
        else:
            status = 'unavailable'
            d_ts = None
            e_ts = None

        orders.append({
            'order_id': f'O{i:05d}',
            'customer_id': cust_id,
            'order_status': status,
            'order_purchase_timestamp': p_ts,
            'order_approved_at': a_ts,
            'order_delivered_customer_date': d_ts,
            'order_estimated_delivery_date': e_ts,
        })
    df_o = pd.DataFrame(orders)
    conn.execute("CREATE OR REPLACE TABLE olist_orders AS SELECT * FROM df_o")

    # Order items
    items = []
    for i in range(1, n_orders + 1):
        oid = f'O{i:05d}'
        n_it = random.choices([1, 2, 3, 4, 5], weights=[60, 25, 10, 3, 2])[0]
        for j in range(1, n_it + 1):
            items.append({
                'order_id': oid,
                'order_item_id': j,
                'product_id': f'P{random.randint(1, n_products):05d}',
                'seller_id': f'S{random.randint(1, n_sellers):05d}',
                'price': round(random.uniform(10, 500), 2),
                'freight_value': round(random.uniform(5, 50), 2),
            })
    df_i = pd.DataFrame(items)
    conn.execute("CREATE OR REPLACE TABLE olist_order_items AS SELECT * FROM df_i")

    # Payments
    payments = []
    for i in range(1, n_orders + 1):
        oid = f'O{i:05d}'
        n_pay = random.choices([1, 2], weights=[90, 10])[0]
        for j in range(1, n_pay + 1):
            payments.append({
                'order_id': oid,
                'payment_sequential': j,
                'payment_type': random.choice(PAYMENT_TYPES),
                'payment_installments': random.choices(
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
                    weights=[30, 15, 15, 10, 10, 5, 5, 3, 3, 2, 2])[0],
                'payment_value': round(random.uniform(10, 500), 2),
            })
    df_pay = pd.DataFrame(payments)
    conn.execute("CREATE OR REPLACE TABLE olist_order_payments AS SELECT * FROM df_pay")

    # Reviews (not strictly needed for prepare_data but included for completeness)
    reviews = []
    for i in range(1, n_orders + 1):
        o = orders[i - 1]
        if o['order_status'] == 'delivered' and random.random() < 0.85:
            is_late = (o['order_delivered_customer_date'] > o['order_estimated_delivery_date']
                       if (o['order_delivered_customer_date'] and o['order_estimated_delivery_date'])
                       else False)
            sw = [0.20, 0.25, 0.30, 0.15, 0.10] if is_late else [0.05, 0.05, 0.10, 0.25, 0.55]
            reviews.append({
                'review_id': f'R{i:05d}',
                'order_id': oid,
                'review_score': random.choices([1, 2, 3, 4, 5], weights=sw)[0],
                'review_comment_title': None,
                'review_comment_message': None,
                'review_creation_date': (o['order_delivered_customer_date']
                                         + timedelta(days=random.randint(0, 7))
                                         if o['order_delivered_customer_date'] else None),
                'review_answer_timestamp': None,
            })
    df_r = pd.DataFrame(reviews)
    conn.execute("CREATE OR REPLACE TABLE olist_order_reviews AS SELECT * FROM df_r")

    conn.close()
    print(f"[01_prepare_data] Database built: {db_path}")
    return db_path


# ---------------------------------------------------------------------------
# 2.  Feature engineering
# ---------------------------------------------------------------------------
def build_features(conn):
    """Return a DataFrame with one row per delivered order."""

    # -- Base order-level features (SQL aggregations) -------------------------
    base = conn.execute("""
        SELECT
            o.order_id,
            o.customer_id,
            o.order_purchase_timestamp,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date,

            o.order_delivered_customer_date > o.order_estimated_delivery_date
                AS target_late,

            c.customer_state,
            c.customer_city,

            -- Item aggregations
            COUNT(DISTINCT oi.order_item_id)                 AS n_items,
            ROUND(SUM(oi.price)::DOUBLE, 2)                  AS total_price,
            ROUND(SUM(oi.freight_value)::DOUBLE, 2)          AS total_freight,
            ROUND(AVG(oi.freight_value / NULLIF(oi.price, 0))::DOUBLE, 4)
                AS freight_to_price_ratio,

            -- Sellers per order (for cross_state logic)
            LIST(DISTINCT oi.seller_id)                      AS seller_ids,

            -- Product categories (pick most common)
            LIST(DISTINCT p.product_category_name)           AS categories,

            -- Seller states (to check cross_state)
            LIST(DISTINCT s.seller_state)                    AS seller_states

        FROM olist_orders o
        JOIN olist_order_items oi  ON o.order_id = oi.order_id
        JOIN olist_products p      ON oi.product_id = p.product_id
        JOIN olist_sellers s       ON oi.seller_id = s.seller_id
        JOIN olist_customers c     ON o.customer_id = c.customer_id

        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL

        GROUP BY o.order_id, o.customer_id, o.order_purchase_timestamp,
                 o.order_delivered_customer_date, o.order_estimated_delivery_date,
                 c.customer_state, c.customer_city
    """).fetchdf()

    print(f"[01_prepare_data] Base orders: {len(base)}")

    # -- Payment features (join, then take first payment per order) -----------
    pmts = conn.execute("""
        SELECT order_id,
               MAX(payment_installments) AS payment_installments,
               LIST(DISTINCT payment_type) AS payment_types
        FROM olist_order_payments
        GROUP BY order_id
    """).fetchdf()

    # Take the first payment type alphabetically as a simple deterministic choice
    pmts['payment_type'] = pmts['payment_types'].apply(
        lambda lst: sorted(list(lst))[0] if lst is not None and len(lst) > 0 else 'unknown'
    )
    pmts.drop(columns=['payment_types'], inplace=True)

    base = base.merge(pmts, on='order_id', how='left')
    base['payment_installments'] = base['payment_installments'].fillna(1)
    base['payment_type'] = base['payment_type'].fillna('unknown')

    # -- Time-based features -------------------------------------------------
    base['order_purchase_timestamp'] = pd.to_datetime(base['order_purchase_timestamp'])
    base['purchase_weekday'] = base['order_purchase_timestamp'].dt.dayofweek   # 0=Mon
    base['purchase_month']   = base['order_purchase_timestamp'].dt.month

    base['promised_days'] = (
        pd.to_datetime(base['order_estimated_delivery_date'])
        - base['order_purchase_timestamp']
    ).dt.days

    # -- Cross-state: 1 if any seller is in a different state than customer ---
    def any_cross_state(row):
        cust_state = row['customer_state']
        seller_states = row['seller_states']
        if isinstance(seller_states, list) and len(seller_states) > 0:
            return 1 if any(s != cust_state for s in seller_states) else 0
        return 0

    base['cross_state'] = base.apply(any_cross_state, axis=1)

    # -- Product category: use the first category alphabetically (deterministic)
    base['product_category'] = base['categories'].apply(
        lambda lst: sorted(list(lst))[0] if lst is not None and len(lst) > 0 else 'unknown'
    )

    # -- Expand seller_ids list into rows for seller-level computations -------
    # We'll keep them for the expanding late-rate feature
    # For now, drop the list columns
    base.drop(columns=['categories', 'seller_states'], inplace=True)

    # -- Sort by purchase timestamp for expanding window ----------------------
    base.sort_values('order_purchase_timestamp', inplace=True)
    base.reset_index(drop=True, inplace=True)

    return base


# ---------------------------------------------------------------------------
# 3.  Seller historical late rate  (expanding window, time-ordered)
# ---------------------------------------------------------------------------
def add_seller_hist_late_rate(df, conn):
    """
    For each order, compute the seller's late rate on ALL strictly prior
    orders of that seller.  If the seller has <3 prior orders, fall back to
    the global running average (over all sellers).
    """
    print("[01_prepare_data] Computing seller_hist_late_rate (expanding window)…")

    # We need seller-level truth per order-item
    seller_order_late = conn.execute("""
        SELECT
            oi.seller_id,
            o.order_id,
            o.order_purchase_timestamp,
            (o.order_delivered_customer_date > o.order_estimated_delivery_date)::INT
                AS is_late
        FROM olist_orders o
        JOIN olist_order_items oi ON o.order_id = oi.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    """).fetchdf()

    seller_order_late.sort_values('order_purchase_timestamp', inplace=True)

    # Build per-order seller late rates
    # Map order_id -> set of seller_ids for that order
    order_sellers = conn.execute("""
        SELECT order_id, LIST(DISTINCT seller_id) AS seller_list
        FROM olist_order_items
        GROUP BY order_id
    """).fetchdf()
    order_seller_map = {}
    for oid, slist in zip(order_sellers['order_id'], order_sellers['seller_list']):
        if isinstance(slist, (list, np.ndarray)):
            order_seller_map[oid] = [str(s) for s in slist]
        else:
            order_seller_map[oid] = [str(slist)]

    # Build expanding stats per seller
    seller_stats = {}        # seller_id -> {'count': int, 'late': int}
    global_count = 0
    global_late = 0

    # We'll store result in a dict: order_id -> seller_hist_late_rate
    result_map = {}

    # Process orders in time order
    for order_id in df['order_id']:
        sellers = order_seller_map.get(order_id, [])
        if not isinstance(sellers, list):
            sellers = [sellers]

        # Compute the rate from prior data
        rates = []
        for sid in sellers:
            stats = seller_stats.get(sid, {'count': 0, 'late': 0})
            if stats['count'] >= 3:
                # Use seller's own rate
                rate = stats['late'] / stats['count']
            else:
                # Fallback to global running rate
                if global_count >= 3:
                    rate = global_late / global_count
                else:
                    rate = 0.0  # not enough history yet
            rates.append(rate)

        # Average across sellers in this order (or take max, etc.)
        # Using mean as a balanced aggregate
        result_map[order_id] = np.mean(rates) if rates else 0.0

        # Now update stats with THIS order's outcome (for future orders)
        for sid in sellers:
            # Find if this seller's item in this order was late
            mask = (seller_order_late['seller_id'] == sid) & (seller_order_late['order_id'] == order_id)
            match = seller_order_late.loc[mask]
            if not match.empty:
                late_val = match.iloc[0]['is_late']
                if sid not in seller_stats:
                    seller_stats[sid] = {'count': 0, 'late': 0}
                seller_stats[sid]['count'] += 1
                seller_stats[sid]['late'] += late_val
                global_count += 1
                global_late += late_val

    df['seller_hist_late_rate'] = df['order_id'].map(result_map)
    return df


# ---------------------------------------------------------------------------
# 4.  Main
# ---------------------------------------------------------------------------
def main():
    db_path = find_or_build_db()
    conn = duckdb.connect(str(db_path))

    print("[01_prepare_data] Building features…")
    df = build_features(conn)

    print("[01_prepare_data] Adding seller historical late rate…")
    df = add_seller_hist_late_rate(df, conn)

    conn.close()

    # -- Select & order final columns ----------------------------------------
    feature_cols = [
        'order_id',
        'target_late',
        'n_items',
        'total_price',
        'total_freight',
        'freight_to_price_ratio',
        'payment_installments',
        'payment_type',
        'promised_days',
        'purchase_weekday',
        'purchase_month',
        'cross_state',
        'product_category',
        'seller_hist_late_rate',
    ]
    df_out = df[feature_cols].copy()
    df_out['target_late'] = df_out['target_late'].astype(int)

    # -- Save -----------------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[01_prepare_data] Saved: {OUTPUT_CSV}")
    print(f"[01_prepare_data] Rows: {len(df_out)}, Columns: {len(df_out.columns)}")

    # -- Target balance -------------------------------------------------------
    n_late = df_out['target_late'].sum()
    n_on_time = len(df_out) - n_late
    pct_late = 100.0 * n_late / len(df_out) if len(df_out) > 0 else 0
    print(f"\n{'=' * 55}")
    print("TARGET BALANCE")
    print(f"{'=' * 55}")
    print(f"  target_late = 1 (late):  {n_late:>5}  ({pct_late:.1f}%)")
    print(f"  target_late = 0 (on-time): {n_on_time:>5}  ({100 - pct_late:.1f}%)")
    print(f"  Total:                  {len(df_out):>5}")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()