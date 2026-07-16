"""
Analysis script for the Olist E-Commerce DuckDB dataset.

Connects to data/olist.duckdb and runs 4-5 SQL queries:

  1) Revenue and order count by customer state [descriptive]
  2) Average delivery days and review score by product category [descriptive]
  3) Late-delivery rate by cross_state_shipment [diagnostic]
  4) Average review score for late vs on-time orders [diagnostic]

Each result is printed as a formatted table with a one-line interpretation
above it.

Usage: python src/04_analysis.py
"""

import sys

try:
    import duckdb
    import pandas as pd
except ImportError:
    print("This script requires duckdb and pandas. Install with: pip install duckdb pandas")
    sys.exit(1)


def fmt_table(df, max_rows=100):
    """Return a string representation of a DataFrame."""
    if df.empty:
        return "(empty)"
    return df.to_string(index=False, max_rows=max_rows)


def main(db_path='data/olist.duckdb'):
    conn = duckdb.connect(database=db_path, read_only=True)

    # Verify the database has the expected tables
    tables = conn.execute("SHOW TABLES").fetchdf()['name'].tolist()
    required = {'olist_orders', 'olist_customers', 'olist_order_items',
                'olist_products', 'olist_sellers', 'olist_order_reviews'}
    missing = required - set(tables)
    if missing:
        print(f"ERROR: Missing tables in {db_path}: {missing}")
        print("Run 'python scripts/build_olist_db.py' first.")
        return

    # =========================================================================
    # Query 1: Revenue and order count by customer state [descriptive]
    # =========================================================================
    print("=" * 72)
    print("QUERY 1 [DESCRIPTIVE] – Revenue & order count by customer state")
    print("=" * 72)
    print("Interpretation: Shows which states generate the most orders and revenue,"
          " identifying top regional markets.")
    print()

    q1 = """
    SELECT
        c.customer_state                        AS customer_state,
        COUNT(DISTINCT o.order_id)               AS order_count,
        ROUND(SUM(oi.price + oi.freight_value)::DOUBLE, 2) AS total_revenue
    FROM olist_orders o
    JOIN olist_customers c  ON o.customer_id = c.customer_id
    JOIN olist_order_items oi ON o.order_id = oi.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_state
    ORDER BY total_revenue DESC
    """
    df1 = conn.execute(q1).fetchdf()
    print(fmt_table(df1))
    if not df1.empty:
        top = df1.iloc[0]
        print(f"\n  >> Top state: {top['customer_state']} with {top['order_count']} orders"
              f" and ${top['total_revenue']:.2f} revenue.\n")

    # =========================================================================
    # Query 2: Avg delivery days and review score by product category [descriptive]
    # =========================================================================
    print("=" * 72)
    print("QUERY 2 [DESCRIPTIVE] – Avg delivery days & review score by product category")
    print("=" * 72)
    print("Interpretation: Identifies product categories with slow delivery"
          " or low satisfaction, flagging areas for operational improvement.")
    print()

    q2 = """
    SELECT
        p.product_category_name                              AS product_category,
        ROUND(AVG(EXTRACT(DAY FROM (o.order_delivered_customer_date
                                     - o.order_purchase_timestamp)))::DOUBLE, 2) AS avg_delivery_days,
        ROUND(AVG(r.review_score)::DOUBLE, 2)                AS avg_review_score,
        COUNT(*)                                             AS n_orders
    FROM olist_orders o
    JOIN olist_order_items oi  ON o.order_id = oi.order_id
    JOIN olist_products p      ON oi.product_id = p.product_id
    JOIN olist_order_reviews r ON o.order_id = r.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    GROUP BY p.product_category_name
    ORDER BY avg_delivery_days DESC
    """
    df2 = conn.execute(q2).fetchdf()
    print(fmt_table(df2.head(20)))
    print(f"  ({len(df2)} categories total, showing top 20 by delivery days)")
    if not df2.empty:
        print(f"\n  >> Fastest: {df2.iloc[-1]['product_category']} ({df2.iloc[-1]['avg_delivery_days']:.1f} days,"
              f" rating={df2.iloc[-1]['avg_review_score']:.2f})")
        print(f"  >> Slowest: {df2.iloc[0]['product_category']} ({df2.iloc[0]['avg_delivery_days']:.1f} days,"
              f" rating={df2.iloc[0]['avg_review_score']:.2f})\n")

    # =========================================================================
    # Query 3: Late-delivery rate by cross_state_shipment [diagnostic]
    # =========================================================================
    print("=" * 72)
    print("QUERY 3 [DIAGNOSTIC] – Late-delivery rate by cross-state vs same-state shipment")
    print("=" * 72)
    print("Interpretation: Determines whether cross-state shipments are more"
          " likely to be late, quantifying the impact of geographic distance on delivery performance.")
    print()

    q3 = """
    WITH shipment_flag AS (
        SELECT
            o.order_id,
            c.customer_state,
            s.seller_state,
            CASE WHEN c.customer_state != s.seller_state THEN 'cross_state' ELSE 'same_state' END AS cross_state_shipment,
            o.order_delivered_customer_date,
            o.order_estimated_delivery_date
        FROM olist_orders o
        JOIN olist_customers c ON o.customer_id = c.customer_id
        JOIN olist_order_items oi ON o.order_id = oi.order_id
        JOIN olist_sellers s ON oi.seller_id = s.seller_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    )
    SELECT
        cross_state_shipment,
        COUNT(*)                                        AS total_orders,
        SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date
                 THEN 1 ELSE 0 END)                     AS late_orders,
        ROUND(100.0 * SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date
                              THEN 1 ELSE 0 END) / COUNT(*), 2) AS late_rate_pct
    FROM shipment_flag
    GROUP BY cross_state_shipment
    ORDER BY late_rate_pct DESC
    """
    df3 = conn.execute(q3).fetchdf()
    print(fmt_table(df3))
    if not df3.empty:
        print(f"\n  >> Cross-state shipments have a {df3.loc[df3['cross_state_shipment']=='cross_state', 'late_rate_pct'].values[0]:.1f}%"
              f" late rate vs {df3.loc[df3['cross_state_shipment']=='same_state', 'late_rate_pct'].values[0]:.1f}% for same-state.\n")

    # =========================================================================
    # Query 4: Avg review score for late vs on-time orders [diagnostic]
    # =========================================================================
    print("=" * 72)
    print("QUERY 4 [DIAGNOSTIC] – Average review score for late vs on-time orders")
    print("=" * 72)
    print("Interpretation: Quantifies the impact of delivery punctuality on customer"
          " satisfaction — late deliveries result in measurably lower review scores.")
    print()

    q4 = """
    WITH delivery_status AS (
        SELECT
            o.order_id,
            CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date
                 THEN 'late' ELSE 'on_time' END AS delivery_type,
            r.review_score
        FROM olist_orders o
        JOIN olist_order_reviews r ON o.order_id = r.order_id
        WHERE o.order_status = 'delivered'
          AND o.order_delivered_customer_date IS NOT NULL
          AND o.order_estimated_delivery_date IS NOT NULL
    )
    SELECT
        delivery_type,
        COUNT(*)                        AS n_orders,
        ROUND(AVG(review_score)::DOUBLE, 2) AS avg_review_score,
        ROUND(STDDEV_SAMP(review_score)::DOUBLE, 2) AS std_review_score
    FROM delivery_status
    GROUP BY delivery_type
    ORDER BY avg_review_score DESC
    """
    df4 = conn.execute(q4).fetchdf()
    print(fmt_table(df4))
    if not df4.empty:
        print(f"\n  >> On-time orders score {df4.loc[df4['delivery_type']=='on_time', 'avg_review_score'].values[0]:.2f} vs"
              f" late orders score {df4.loc[df4['delivery_type']=='late', 'avg_review_score'].values[0]:.2f}"
              f" — a gap of {abs(df4['avg_review_score'].diff().iloc[-1]):.2f} points.\n")

    conn.close()
    print("Analysis complete.")


if __name__ == '__main__':
    main()