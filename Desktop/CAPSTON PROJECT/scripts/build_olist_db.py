"""
Build a synthetic Olist-like DuckDB database for analysis.

Creates tables mimicking the Brazilian E-Commerce Olist dataset:
  - olist_customers
  - olist_orders
  - olist_order_items
  - olist_products
  - olist_sellers
  - olist_order_reviews
  - olist_order_payments

Usage: python scripts/build_olist_db.py
"""

import duckdb
import random
import pandas as pd
from datetime import datetime, timedelta
import os

random.seed(42)

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "olist.duckdb")

# Brazilian states
STATES = ['SP', 'RJ', 'MG', 'RS', 'PR', 'SC', 'BA', 'DF', 'GO', 'ES',
          'PE', 'CE', 'PA', 'MT', 'MS', 'MA', 'PB', 'AL', 'RN', 'PI',
          'SE', 'RO', 'TO', 'AC', 'AM', 'RR', 'AP']

CITIES_BY_STATE = {
    'SP': ['sao paulo', 'campinas', 'guarulhos', 'santos', 'ribeirao preto'],
    'RJ': ['rio de janeiro', 'niteroi', 'duque de caxias', 'nova iguaçu', 'campos'],
    'MG': ['belo horizonte', 'uberlandia', 'contagem', 'juiz de fora', 'montes claros'],
    'RS': ['porto alegre', 'caxias do sul', 'pelotas', 'canoas', 'santa maria'],
    'PR': ['curitiba', 'londrina', 'maringa', 'ponta grossa', 'cascavel'],
    'SC': ['florianopolis', 'joinville', 'blumenau', 'sao jose', 'criciuma'],
    'BA': ['salvador', 'feira de santana', 'vitoria da conquista', 'ilheus', 'itabuna'],
    'DF': ['brasilia', 'taguatinga', 'ceilandia', 'guara', 'samambaia'],
    'GO': ['goiania', 'aparecida de goiania', 'anapolis', 'rio verde', 'luziania'],
    'ES': ['vitoria', 'vila velha', 'serra', 'cariacica', 'linhares'],
    'PE': ['recife', 'olinda', 'jaboatao', 'caruaru', 'petrolina'],
    'CE': ['fortaleza', 'caucaia', 'juazeiro do norte', 'maracanau', 'sobral'],
    'PA': ['belem', 'anamideua', 'santarem', 'maraba', 'castanhal'],
    'MT': ['cuiaba', 'varzea grande', 'rondonopolis', 'sinop', 'tangara da serra'],
    'MS': ['campo grande', 'dourados', 'três lagoas', 'corumba', 'ponta pora'],
    'MA': ['sao luis', 'imperatriz', 'sao jose de ribamar', 'caxias', 'timon'],
    'PB': ['joao pessoa', 'campina grande', 'santa rita', 'patos', 'bayeux'],
    'AL': ['maceio', 'arapiraca', 'rio largo', 'penedo', 'delmiro gouveia'],
    'RN': ['natal', 'mossoro', 'parnamirim', 'sao goncalo do amarante', 'caico'],
    'PI': ['teresina', 'parnaiba', 'picos', 'floriano', 'campo maior'],
    'SE': ['aracaju', 'nossa senhora do socorro', 'lagarto', 'itabaiana', 'estancia'],
    'RO': ['porto velho', 'ji-parana', 'ariquemes', 'vilhena', 'cacoal'],
    'TO': ['palmas', 'araguaina', 'gurupi', 'porto nacional', 'paraíso do tocantins'],
    'AC': ['rio branco', 'cruzeiro do sul', 'sena madureira', 'tarauaca', 'feijo'],
    'AM': ['manaus', 'parintins', 'itacoatiara', 'manacapuru', 'coari'],
    'RR': ['boa vista', 'rorainopolis', 'caracarai', 'mucajai', 'alto alegre'],
    'AP': ['macapa', 'santana', 'laranjal do jari', 'oida', 'porto grande']
}

PRODUCT_CATEGORIES = [
    'beleza_saude', 'moveis_decoracao', 'informatica_acessorios',
    'cama_mesa_banho', 'esporte_lazer', 'telefonia',
    'relogios_presentes', 'automotivo', 'brinquedos',
    'ferramentas_jardim', 'eletrodomesticos', 'livros_interesse',
    'eletronicos', 'alimentos_bebidas', 'bebes',
    'papelaria', 'moda_roupa', 'casa_conforto',
    'perfumaria', 'utilidades_domesticas', 'consoles_games',
    'musica_arte', 'malas_acessorios', 'moveis_escritorio',
    'construcao_ferramentas', 'audio', 'cool_stuff',
    'portateis_casa_forno_cafe', 'moveis_sala', 'fashion_bolsas',
    'fashion_calcados', 'fashion_roupa_feminina', 'fashion_roupa_masculina',
    'fashion_underwear', 'eletroportateis', 'instrumentos_musicais',
    'artes', 'casa_conforto_2', 'moveis_cozinha', 'moveis_quarto',
    'sinalizacao_seguranca', 'casa_construcao', 'flores',
    'cds_dvds_musicais', 'dvds_blu_ray', 'fashion_esporte',
    'la_cuisine', 'market_place', 'artigos_nautica',
    'eletrodomesticos_2', 'fraldas_higiene', 'moveis_jardim',
    'pc_gamer', 'portateis_cozinha', 'seguros_servicos',
    'casa_limpeza', 'moveis_colchao_estofado', 'carnes',
    'moveis_presentes', 'casa_utilidades', 'casa_iluminacao',
    'casa_escritorio', 'casa_banheiro', 'casa_cozinha',
    'casa_quarto', 'casa_sala', 'casa_jardim',
    'casa_lavanderia', 'casa_garagem', 'casa_outros'
]

PAYMENT_TYPES = ['credit_card', 'boleto', 'voucher', 'debit_card']


def random_date(start, end):
    """Return a random datetime between start and end."""
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def main():
    print("Building Olist DuckDB database...")
    conn = duckdb.connect(str(DB_PATH))

    # ----------------------------------------------------------------
    # 1. CUSTOMERS
    # ----------------------------------------------------------------
    n_customers = 2000
    customers = []
    for i in range(1, n_customers + 1):
        state = random.choice(STATES)
        city = random.choice(CITIES_BY_STATE[state])
        customers.append({
            'customer_id': f'C{i:05d}',
            'customer_unique_id': f'U{random.randint(10000,99999):05d}',
            'customer_zip_code_prefix': random.randint(10000, 99999),
            'customer_city': city,
            'customer_state': state
        })
    df_customers = pd.DataFrame(customers)
    conn.execute("CREATE OR REPLACE TABLE olist_customers AS SELECT * FROM df_customers")
    print(f"  Created olist_customers: {n_customers} rows")

    # ----------------------------------------------------------------
    # 2. SELLERS
    # ----------------------------------------------------------------
    n_sellers = 300
    sellers = []
    for i in range(1, n_sellers + 1):
        state = random.choice(STATES)
        city = random.choice(CITIES_BY_STATE[state])
        sellers.append({
            'seller_id': f'S{i:05d}',
            'seller_zip_code_prefix': random.randint(10000, 99999),
            'seller_city': city,
            'seller_state': state
        })
    df_sellers = pd.DataFrame(sellers)
    conn.execute("CREATE OR REPLACE TABLE olist_sellers AS SELECT * FROM df_sellers")
    print(f"  Created olist_sellers: {n_sellers} rows")

    # ----------------------------------------------------------------
    # 3. PRODUCTS
    # ----------------------------------------------------------------
    n_products = 500
    products = []
    for i in range(1, n_products + 1):
        cat = random.choice(PRODUCT_CATEGORIES)
        products.append({
            'product_id': f'P{i:05d}',
            'product_category_name': cat,
            'product_name_length': random.randint(20, 80),
            'product_description_length': random.randint(100, 1000),
            'product_photos_qty': random.randint(1, 8),
            'product_weight_g': random.randint(100, 5000),
            'product_length_cm': random.randint(10, 100),
            'product_height_cm': random.randint(5, 50),
            'product_width_cm': random.randint(10, 60)
        })
    df_products = pd.DataFrame(products)
    conn.execute("CREATE OR REPLACE TABLE olist_products AS SELECT * FROM df_products")
    print(f"  Created olist_products: {n_products} rows")

    # ----------------------------------------------------------------
    # 4. ORDERS
    # ----------------------------------------------------------------
    n_orders = 5000
    orders = []
    start_date = datetime(2016, 9, 1)
    end_date = datetime(2018, 10, 31)

    for i in range(1, n_orders + 1):
        customer_id = f'C{random.randint(1, n_customers):05d}'
        purchase_ts = random_date(start_date, end_date)
        approved_ts = purchase_ts + timedelta(hours=random.randint(1, 48))
        status_roll = random.random()
        if status_roll < 0.90:
            order_status = 'delivered'
            delivery_days = random.randint(3, 30)
            delivered_ts = purchase_ts + timedelta(days=delivery_days)
            estimated_ts = purchase_ts + timedelta(days=random.randint(5, 25))
        elif status_roll < 0.95:
            order_status = 'shipped'
            delivered_ts = None
            estimated_ts = purchase_ts + timedelta(days=random.randint(5, 25))
        elif status_roll < 0.98:
            order_status = 'canceled'
            delivered_ts = None
            estimated_ts = None
        else:
            order_status = 'unavailable'
            delivered_ts = None
            estimated_ts = None

        orders.append({
            'order_id': f'O{i:05d}',
            'customer_id': customer_id,
            'order_status': order_status,
            'order_purchase_timestamp': purchase_ts,
            'order_approved_at': approved_ts,
            'order_delivered_customer_date': delivered_ts,
            'order_estimated_delivery_date': estimated_ts
        })
    df_orders = pd.DataFrame(orders)
    conn.execute("CREATE OR REPLACE TABLE olist_orders AS SELECT * FROM df_orders")
    print(f"  Created olist_orders: {n_orders} rows")

    # ----------------------------------------------------------------
    # 5. ORDER ITEMS
    # ----------------------------------------------------------------
    order_items = []
    for i in range(1, n_orders + 1):
        order_id = f'O{i:05d}'
        n_items = random.choices([1, 2, 3, 4, 5], weights=[60, 25, 10, 3, 2])[0]
        for j in range(1, n_items + 1):
            product_id = f'P{random.randint(1, n_products):05d}'
            seller_id = f'S{random.randint(1, n_sellers):05d}'
            price = round(random.uniform(10, 500), 2)
            freight = round(random.uniform(5, 50), 2)
            order_items.append({
                'order_id': order_id,
                'order_item_id': j,
                'product_id': product_id,
                'seller_id': seller_id,
                'price': price,
                'freight_value': freight
            })
    df_items = pd.DataFrame(order_items)
    conn.execute("CREATE OR REPLACE TABLE olist_order_items AS SELECT * FROM df_items")
    print(f"  Created olist_order_items: {len(order_items)} rows")

    # ----------------------------------------------------------------
    # 6. ORDER PAYMENTS
    # ----------------------------------------------------------------
    payments = []
    for i in range(1, n_orders + 1):
        order_id = f'O{i:05d}'
        n_payments = random.choices([1, 2], weights=[90, 10])[0]
        for j in range(1, n_payments + 1):
            ptype = random.choice(PAYMENT_TYPES)
            installments = random.choices([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
                                          weights=[30, 15, 15, 10, 10, 5, 5, 3, 3, 2, 2])[0]
            value = round(random.uniform(10, 500), 2)
            payments.append({
                'order_id': order_id,
                'payment_sequential': j,
                'payment_type': ptype,
                'payment_installments': installments,
                'payment_value': value
            })
    df_payments = pd.DataFrame(payments)
    conn.execute("CREATE OR REPLACE TABLE olist_order_payments AS SELECT * FROM df_payments")
    print(f"  Created olist_order_payments: {len(payments)} rows")

    # ----------------------------------------------------------------
    # 7. ORDER REVIEWS
    # ----------------------------------------------------------------
    reviews = []
    for i in range(1, n_orders + 1):
        order_id = f'O{i:05d}'
        order_status = orders[i-1]['order_status']
        if order_status == 'delivered' and random.random() < 0.85:
            delivered = orders[i-1]['order_delivered_customer_date']
            estimated = orders[i-1]['order_estimated_delivery_date']
            is_late = delivered > estimated if (delivered and estimated) else False
            if is_late:
                score_weights = [0.20, 0.25, 0.30, 0.15, 0.10]
            else:
                score_weights = [0.05, 0.05, 0.10, 0.25, 0.55]
            score = random.choices([1, 2, 3, 4, 5], weights=score_weights)[0]
            reviews.append({
                'review_id': f'R{i:05d}',
                'order_id': order_id,
                'review_score': score,
                'review_comment_title': None,
                'review_comment_message': None,
                'review_creation_date': delivered + timedelta(days=random.randint(0, 7)) if delivered else None,
                'review_answer_timestamp': None
            })
    df_reviews = pd.DataFrame(reviews)
    conn.execute("CREATE OR REPLACE TABLE olist_order_reviews AS SELECT * FROM df_reviews")
    print(f"  Created olist_order_reviews: {len(reviews)} rows")

    # ----------------------------------------------------------------
    # Verify
    # ----------------------------------------------------------------
    tables = conn.execute("SHOW TABLES").fetchdf()['name'].tolist()
    print(f"\nDatabase built successfully. Tables: {tables}")
    for t in tables:
        cnt = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
        print(f"  {t}: {cnt} rows")

    conn.close()
    print(f"\nDatabase saved to: {DB_PATH}")


if __name__ == '__main__':
    main()