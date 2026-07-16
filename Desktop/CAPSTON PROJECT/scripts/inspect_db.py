import duckdb

conn = duckdb.connect('data/olist.duckdb')
tables = conn.execute("SELECT table_name, table_schema FROM information_schema.tables").fetchdf()
print("=== TABLES ===")
print(tables.to_string())

for t in tables['table_name'].tolist():
    print(f"\n=== COLUMNS for {t} ===")
    cols = conn.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{t}'").fetchdf()
    print(cols.to_string())

conn.close()