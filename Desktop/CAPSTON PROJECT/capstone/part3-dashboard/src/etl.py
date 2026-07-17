"""
ETL: Build a lightweight cleaned Olist dataset for the dashboard.

Generates a synthetic Olist DuckDB (if not found) and produces a
cleaned CSV at data/cleaned/orders_clean.csv with all fields needed
for dashboard queries: order-level + customer state + review score +
late flag + product category + delivery stats.

Standalone — does NOT import from Part 1 or Part 2.

Usage: python src/etl.py
"""

import os
import duckdb
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
PART3 = os.path.normpath(os.path.join(HERE, ".."))
DATA_DIR = os.path.join(PART3, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

OUTPUT_CSV = os.path.join(CLEANED_DIR, "orders_clean.csv")

DB_CANDIDATES = [
    os.path.join(PART3, "data", "olist.duckdb"),
    os.path.join(DATA_DIR, "olist.duckdb"),
    os.path.join(os.path.normpath(os.path.join(HERE, "..", "..", "..")), "data", "olist.duckdb"),
]


# ---------------------------------------------------------------------------
# Build / find database
# ---------------------------------------------------------------------------
def _synthetic_build():
    """Generate synthetic Olist tables and return path to db."""
    db_path = os.path.join(PART3, "data", "olist.duckdb")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = duckdb.connect(str(db_path))

    STATES = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'ES',
              'PE', 'CE', 'PA', 'MT', 'MS', 'MA', 'PB', 'AL', 'RN', 'PI',
              'SE', 'RO', 'TO', 'AC', 'AM', 'RR', 'AP']
    CITIES_BY_STATE = {s: [f"city_{s}_{i}" for i in range(1, 6)] for s in STATES}
    CATEGORIES = [
        'beleza_saude', 'moveis_decoracao', 'informatica_acessorios',
        'cama_mesa_banho', 'esporte_lazer', 'telefonia',
        'relogios_presentes', 'automotivo', 'brinquedos',
        'ferramentas_jardim', 'eletrodomesticos', 'livros_interesse',
        'eletronicos', 'alimentos_bebidas', 'bebes',
        'papelaria', 'moda_roupa', 'casa_conforto',
    ]
    PAYMENT_TYPES = ['credit_card', 'boleto', 'voucher', 'debit_card']

    def rand_date(start, end):
        delta = end - start
        return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))

    n_customers = 2000
    customers = []
    for i in range(1, n_customers + 1):
        st = random.choice(STATES)
        customers.append({
            'customer_id': f'C{i:05d}', 'customer_unique_id': f'U{random.randint(10000,99999):05d}',
            'customer_zip_code_prefix': random.randint(10000, 99999),
            'customer_city': random.choice(CITIES_BY_STATE[st]), 'customer_state': st,
        })
    _df = pd.DataFrame(customers)
    conn.execute("CREATE OR REPLACE TABLE olist_customers AS SELECT * FROM _df")

    n_sellers = 300
    sellers = []
    for i in range(1, n_sellers + 1):
        st = random.choice(STATES)
        sellers.append({'seller_id': f'S{i:05d}',
                        'seller_zip_code_prefix': random.randint(10000, 99999),
                        'seller_city': random.choice(CITIES_BY_STATE[st]),
                        'seller_state': st})
    _df = pd.DataFrame(sellers)
    conn.execute("CREATE OR REPLACE TABLE olist_sellers AS SELECT * FROM _df")

    n_products = 500
    products = []
    for i in range(1, n_products + 1):
        products.append({'product_id': f'P{i:05d}',
                         'product_category_name': random.choice(CATEGORIES),
                         'product_name_length': random.randint(20, 80),
                         'product_description_length': random.randint(100, 1000),
                         'product_photos_qty': random.randint(1, 8),
                         'product_weight_g': random.randint(100, 5000),
                         'product_length_cm': random.randint(10, 100),
                         'product_height_cm': random.randint(5, 50),
                         'product_width_cm': random.randint(10, 60)})
    _df = pd.DataFrame(products)
    conn.execute("CREATE OR REPLACE TABLE olist_products AS SELECT * FROM _df")

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
            status = 'shipped'; d_ts = None; e_ts = p_ts + timedelta(days=random.randint(5, 25))
        elif roll < 0.98:
            status = 'canceled'; d_ts = None; e_ts = None
        else:
            status = 'unavailable'; d_ts = None; e_ts = None
        orders.append({'order_id': f'O{i:05d}', 'customer_id': cust_id, 'order_status': status,
                       'order_purchase_timestamp': p_ts, 'order_approved_at': a_ts,
                       'order_delivered_customer_date': d_ts, 'order_estimated_delivery_date': e_ts})
    _df = pd.DataFrame(orders)
    conn.execute("CREATE OR REPLACE TABLE olist_orders AS SELECT * FROM _df")

    items = []
    for i in range(1, n_orders + 1):
        oid = f'O{i:05d}'
        for j in range(1, random.choices([1, 2, 3, 4, 5], weights=[60, 25, 10, 3, 2])[0] + 1):
            items.append({'order_id': oid, 'order_item_id': j,
                          'product_id': f'P{random.randint(1, n_products):05d}',
                          'seller_id': f'S{random.randint(1, n_sellers):05d}',
                          'price': round(random.uniform(10, 500), 2),
                          'freight_value': round(random.uniform(5, 50), 2)})
    _df = pd.DataFrame(items)
    conn.execute("CREATE OR REPLACE TABLE olist_order_items AS SELECT * FROM _df")

    payments = []
    for i in range(1, n_orders + 1):
        oid = f'O{i:05d}'
        for j in range(1, random.choices([1, 2], weights=[90, 10])[0] + 1):
            payments.append({'order_id': oid, 'payment_sequential': j,
                             'payment_type': random.choice(PAYMENT_TYPES),
                             'payment_installments': random.choices(
                                 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
                                 weights=[30, 15, 15, 10, 10, 5, 5, 3, 3, 2, 2])[0],
                             'payment_value': round(random.uniform(10, 500), 2)})
    _df = pd.DataFrame(payments)
    conn.execute("CREATE OR REPLACE TABLE olist_order_payments AS SELECT * FROM _df")

    reviews = []
    for o in orders:
        if o['order_status'] == 'delivered' and random.random() < 0.85:
            is_late = (o['order_delivered_customer_date'] > o['order_estimated_delivery_date']
                       if (o['order_delivered_customer_date'] and o['order_estimated_delivery_date']) else False)
            sw = [0.20, 0.25, 0.30, 0.15, 0.10] if is_late else [0.05, 0.05, 0.10, 0.25, 0.55]
            reviews.append({'review_id': f'R{random.randint(1,99999):05d}', 'order_id': o['order_id'],
                            'review_score': random.choices([1, 2, 3, 4, 5], weights=sw)[0],
                            'review_comment_title': None, 'review_comment_message': None,
                            'review_creation_date': (o['order_delivered_customer_date']
                                                     + timedelta(days=random.randint(0, 7))
                                                     if o['order_delivered_customer_date'] else None),
                            'review_answer_timestamp': None})
    _df = pd.DataFrame(reviews)
    conn.execute("CREATE OR REPLACE TABLE olist_order_reviews AS SELECT * FROM _df")

    conn.close()
    print(f"  Built synthetic database: {db_path}")
    return db_path


def find_or_build_db():
    for p in DB_CANDIDATES:
        if os.path.isfile(p):
            print(f"  Found database: {p}")
            return p
    return _synthetic_build()


# ---------------------------------------------------------------------------
# Main ETL
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("PART 3 — ETL")
    print("=" * 60)

    db_path = find_or_build_db()
    conn = duckdb.connect(str(db_path))

    print("\nBuilding cleaned orders dataset…")

    query = """
    SELECT
        o.order_id,
        o.customer_id,
        c.customer_state,
        c.customer_city,
        o.order_status,
        o.order_purchase_timestamp,
        o.order_approved_at,
        o.order_delivered_customer_date,
        o.order_estimated_delivery_date,

        CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
             THEN 1 ELSE 0 END AS is_late,

        EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_purchase_timestamp))::INT
            AS delivery_days,

        EXTRACT(DAY FROM (o.order_estimated_delivery_date - o.order_purchase_timestamp))::INT
            AS promised_days,

        COUNT(DISTINCT oi.order_item_id) AS n_items,
        ROUND(SUM(oi.price)::DOUBLE, 2) AS total_price,
        ROUND(SUM(oi.freight_value)::DOUBLE, 2) AS total_freight,
        ROUND(AVG(oi.freight_value / NULLIF(oi.price, 0))::DOUBLE, 4) AS freight_to_price_ratio,

        MAX(CASE WHEN c.customer_state != s.seller_state THEN 1 ELSE 0 END) AS cross_state,

        MIN(p.product_category_name) AS product_category,

        AVG(r.review_score) AS avg_review_score

    FROM olist_orders o
    JOIN olist_customers c ON o.customer_id = c.customer_id
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    JOIN olist_products p ON oi.product_id = p.product_id
    JOIN olist_sellers s ON oi.seller_id = s.seller_id
    LEFT JOIN olist_order_reviews r ON o.order_id = r.order_id

    GROUP BY o.order_id, o.customer_id, c.customer_state, c.customer_city,
             o.order_status, o.order_purchase_timestamp, o.order_approved_at,
             o.order_delivered_customer_date, o.order_estimated_delivery_date,
             o.order_delivered_customer_date > o.order_estimated_delivery_date,
             EXTRACT(DAY FROM (o.order_delivered_customer_date - o.order_purchase_timestamp)),
             EXTRACT(DAY FROM (o.order_estimated_delivery_date - o.order_purchase_timestamp))
    """

    df = conn.execute(query).fetchdf()
    conn.close()

    df['avg_review_score'] = df['avg_review_score'].fillna(0)

    print(f"  Orders extracted: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Late orders: {df['is_late'].sum()} / {len(df)} ({100*df['is_late'].sum()/len(df):.1f}%)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved to: {OUTPUT_CSV}")
    print("ETL complete.")


if __name__ == '__main__':
    main()