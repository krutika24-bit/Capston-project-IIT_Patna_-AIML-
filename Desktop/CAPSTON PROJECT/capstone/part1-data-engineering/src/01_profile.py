import pandas as pd
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "Womens Clothing E-Commerce Reviews.csv")

if not os.path.exists(DATA_FILE):
    print("File not found:", DATA_FILE)
    exit(1)

filepath = DATA_FILE
filename = os.path.basename(filepath)

print("=" * 70)
print(f"FILE: {filename}")
print("=" * 70)

df = pd.read_csv(filepath)

# Shape
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

# Dtypes
print(f"\nColumn dtypes:")
for col, dtype in df.dtypes.items():
    print(f"  {col}: {dtype}")

# % Missing per column
print(f"\n% Missing per column:")
missing_pct = (df.isnull().sum() / len(df)) * 100
for col, pct in missing_pct.items():
    print(f"  {col}: {pct:.2f}%")

# Duplicate row count
dup_count = df.duplicated().sum()
print(f"\nDuplicate rows: {dup_count} ({dup_count / len(df) * 100:.2f}%)")

# Value counts for categorical columns
cat_cols = ['Division Name', 'Department Name', 'Class Name', 'Rating']
for col in cat_cols:
    if col in df.columns:
        print(f"\nValue counts for '{col}':")
        vc = df[col].value_counts()
        for val, cnt in vc.items():
            print(f"  {val}: {cnt}")
print()