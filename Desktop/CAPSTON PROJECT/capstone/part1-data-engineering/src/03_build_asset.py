import pandas as pd
import duckdb
import os

# Paths
CLEANED_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "womens_clothing_reviews_clean.csv")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_PATH = os.path.join(DB_DIR, "reviews.duckdb")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "..", "schema.sql")

print("=" * 70)
print("BUILDING DUCKDB ASSET")
print("=" * 70)

# Load cleaned data
print(f"\nLoading cleaned data from: {CLEANED_FILE}")
df = pd.read_csv(CLEANED_FILE)
print(f"  Loaded {len(df)} rows x {len(df.columns)} columns")

# Add derived columns
print("\nAdding derived columns...")

# review_length: word count of review_text
df['review_length'] = df['review_text'].str.split().str.len()
print(f"  'review_length' added (word count of review_text)")
print(f"    Min: {df['review_length'].min()}, Max: {df['review_length'].max()}, Mean: {df['review_length'].mean():.1f}")

# age_bucket: categorize age into buckets
def age_bucket(age):
    if age < 25:
        return '<25'
    elif age <= 34:
        return '25-34'
    elif age <= 44:
        return '35-44'
    elif age <= 54:
        return '45-54'
    else:
        return '55+'

df['age_bucket'] = df['age'].apply(age_bucket)
print(f"  'age_bucket' added")
age_dist = df['age_bucket'].value_counts().sort_index()
for bucket, cnt in age_dist.items():
    print(f"    {bucket}: {cnt}")

print(f"\nFinal shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Final columns: {list(df.columns)}")

# Write schema.sql
print(f"\nWriting schema to: {SCHEMA_PATH}")
schema_lines = [
    "-- ============================================================",
    "-- Schema for reviews table (DuckDB)",
    "-- Source: Women's Clothing E-Commerce Reviews",
    "-- ============================================================",
    "",
    "-- Column descriptions and types",
    "",
    "/*",
]
for col in df.columns:
    dtype = df[col].dtype
    if dtype == 'int64':
        sql_type = 'INTEGER'
    elif dtype == 'float64':
        sql_type = 'FLOAT'
    else:
        sql_type = 'VARCHAR'
    
    col_desc = ""
    if col == 'clothing_id':
        col_desc = "Unique identifier for the clothing item"
    elif col == 'age':
        col_desc = "Age of the reviewer (clipped to 18-100)"
    elif col == 'title':
        col_desc = "Title of the review (may be null)"
    elif col == 'review_text':
        col_desc = "Text content of the review"
    elif col == 'rating':
        col_desc = "Rating given by the reviewer (1 to 5)"
    elif col == 'recommended_ind':
        col_desc = "Whether the reviewer recommends the product (0/1)"
    elif col == 'positive_feedback_count':
        col_desc = "Number of positive feedback votes on the review"
    elif col == 'division_name':
        col_desc = "Product division (General, General Petite, Initmates)"
    elif col == 'department_name':
        col_desc = "Product department (Tops, Dresses, Bottoms, etc.)"
    elif col == 'class_name':
        col_desc = "Product class (Dresses, Knits, Blouses, etc.)"
    elif col == 'review_length':
        col_desc = "Derived: word count of review_text"
    elif col == 'age_bucket':
        col_desc = "Derived: age category bucket (<25, 25-34, 35-44, 45-54, 55+)"
    
    schema_lines.append(f"  {col}: {sql_type}  -- {col_desc}")

schema_lines.extend([
    "*/",
    "",
    "-- DuckDB CREATE TABLE statement",
    "CREATE TABLE IF NOT EXISTS reviews (",
])
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    sql_type = 'INTEGER' if dtype == 'int64' else ('FLOAT' if dtype == 'float64' else 'VARCHAR')
    comma = "," if i < len(df.columns) - 1 else ""
    schema_lines.append(f"    {col} {sql_type}{comma}")
schema_lines.append(");")

with open(SCHEMA_PATH, 'w') as f:
    f.write('\n'.join(schema_lines))
print("  Done.")

# Write to DuckDB
print(f"\nWriting to DuckDB: {DB_PATH}")
os.makedirs(DB_DIR, exist_ok=True)
con = duckdb.connect(str(DB_PATH))
con.execute("CREATE TABLE IF NOT EXISTS reviews AS SELECT * FROM df")
row_count = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
print(f"  Table 'reviews' created with {row_count} rows")
con.close()

print(f"\n{'=' * 70}")
print("BUILD COMPLETE")
print(f"{'=' * 70}")
print(f"  DuckDB file: {DB_PATH}")
print(f"  Schema file: {SCHEMA_PATH}")
print(f"  Table: reviews ({row_count} rows, {len(df.columns)} columns)")