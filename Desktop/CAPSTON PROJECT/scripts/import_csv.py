import duckdb
import os
print('Importing CSV into data/reviews.duckdb...')
db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'reviews.duckdb')
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Womens Clothing E-Commerce Reviews.csv')
conn = duckdb.connect(database=db_path)
conn.execute("CREATE TABLE IF NOT EXISTS reviews AS SELECT * FROM read_csv_auto('%s', AUTO_DETECT=TRUE)" % csv_path.replace('\\','/'))
print('Import complete. Tables:')
print(conn.execute('SHOW TABLES').fetchdf())
conn.close()
