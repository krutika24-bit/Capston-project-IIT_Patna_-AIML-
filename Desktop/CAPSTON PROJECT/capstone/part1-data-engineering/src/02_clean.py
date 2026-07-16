import pandas as pd
import os

DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "Womens Clothing E-Commerce Reviews.csv")

if not os.path.exists(DATA_FILE):
    print("File not found:", DATA_FILE)
    exit(1)

filepath = DATA_FILE
filename = os.path.basename(filepath)

print("=" * 70)
print(f"LOADING: {filename}")
print("=" * 70)

df = pd.read_csv(filepath)
print(f"Initial rows: {len(df)}")

# ---------------------------------------------------------------------------
# Step 1: Drop the unnamed index column
# ---------------------------------------------------------------------------
print("\n--- Step 1: Drop unnamed index column ---")
before = len(df.columns)
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])
    print(f"  Dropped 'Unnamed: 0' column. Columns: {before} -> {len(df.columns)}")
else:
    print("  'Unnamed: 0' column not found, skipping.")

# ---------------------------------------------------------------------------
# Step 2: Rename columns to snake_case
# ---------------------------------------------------------------------------
print("\n--- Step 2: Rename columns to snake_case ---")
rename_map = {
    'Clothing ID': 'clothing_id',
    'Age': 'age',
    'Title': 'title',
    'Review Text': 'review_text',
    'Rating': 'rating',
    'Recommended IND': 'recommended_ind',
    'Positive Feedback Count': 'positive_feedback_count',
    'Division Name': 'division_name',
    'Department Name': 'department_name',
    'Class Name': 'class_name',
}
df = df.rename(columns=rename_map)
print(f"  Columns renamed: {list(df.columns)}")

# ---------------------------------------------------------------------------
# Step 3: Drop rows with null Review Text
# ---------------------------------------------------------------------------
print("\n--- Step 3: Drop rows with null Review Text ---")
before = len(df)
null_review = df['review_text'].isnull().sum()
df = df.dropna(subset=['review_text'])
after = len(df)
print(f"  Rows with null review_text: {null_review}")
print(f"  Rows: {before} -> {after} (dropped {before - after})")

# ---------------------------------------------------------------------------
# Step 4: Handle nulls in Division Name / Department Name / Class Name
# ---------------------------------------------------------------------------
# Decision: Only 0.06% (~14 rows) have nulls in these categorical columns.
# Since these are nominal categories with no natural "default", and the
# proportion is negligible, we drop those rows rather than imputing.
print("\n--- Step 4: Handle nulls in division_name / department_name / class_name ---")
before = len(df)
null_counts = df[['division_name', 'department_name', 'class_name']].isnull().sum()
print(f"  Null counts before:")
for col, cnt in null_counts.items():
    print(f"    {col}: {cnt}")
df = df.dropna(subset=['division_name', 'department_name', 'class_name'])
after = len(df)
print(f"  Rows: {before} -> {after} (dropped {before - after})")

# ---------------------------------------------------------------------------
# Step 5: Drop exact duplicate rows
# ---------------------------------------------------------------------------
print("\n--- Step 5: Drop exact duplicate rows ---")
before = len(df)
dup_count = df.duplicated().sum()
df = df.drop_duplicates()
after = len(df)
print(f"  Duplicate rows found: {dup_count}")
print(f"  Rows: {before} -> {after} (dropped {before - after})")

# ---------------------------------------------------------------------------
# Step 6: Validate Rating (1-5) and Age (positive integer)
# ---------------------------------------------------------------------------
print("\n--- Step 6: Validate Rating and Age ---")

# Rating validation
before = len(df)
rating_outliers = df[(df['rating'] < 1) | (df['rating'] > 5)]
print(f"  Rating < 1 or > 5: {len(rating_outliers)} rows")
if len(rating_outliers) > 0:
    print(f"    Outlier values: {rating_outliers['rating'].value_counts().to_dict()}")
    # Clip to valid range
    df['rating'] = df['rating'].clip(1, 5)
    print(f"    Clipped rating to [1, 5] range.")

# Age validation
age_outliers = df[(df['age'] <= 0) | (df['age'] > 120)]
print(f"  Age <= 0 or > 120: {len(age_outliers)} rows")
if len(age_outliers) > 0:
    print(f"    Outlier values: {age_outliers['age'].value_counts().to_dict()}")
    # Clip to a reasonable range (18-100 for clothing reviews)
    df['age'] = df['age'].clip(18, 100)
    print(f"    Clipped age to [18, 100] range.")

after = len(df)
print(f"  Rows after validation: {after} (no rows dropped, only clipped)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)
print(f"Final shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Final columns: {list(df.columns)}")
print(f"Final dtypes:\n{df.dtypes}")
print(f"Remaining nulls:\n{df.isnull().sum()}")
print(f"Rating range: {df['rating'].min()} - {df['rating'].max()}")
print(f"Age range: {df['age'].min()} - {df['age'].max()}")

# Save cleaned data
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, "womens_clothing_reviews_clean.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved cleaned data to: {out_path}")