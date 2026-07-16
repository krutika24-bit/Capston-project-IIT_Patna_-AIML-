import duckdb
conn = duckdb.connect(r'C:/Users/HP/Desktop/CAPSTON PROJECT/data/reviews.duckdb')
info = conn.execute("PRAGMA table_info('reviews')").fetchdf()
print(info)
print('\nColumns list:')
print(info['name'].tolist())
conn.close()
