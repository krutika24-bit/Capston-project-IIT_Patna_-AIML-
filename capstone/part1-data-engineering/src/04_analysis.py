import duckdb
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reviews.duckdb")

con = duckdb.connect(str(DB_PATH))

print("=" * 70)
print("ANALYSIS OF WOMEN'S CLOTHING E-COMMERCE REVIEWS")
print("=" * 70)

# -----------------------------------------------------------------------
# Query 1: Average rating and recommend-rate by Department Name
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("QUERY 1: Average Rating & Recommend Rate by Department [Descriptive]")
print("=" * 70)

q1 = """
SELECT
    department_name,
    ROUND(AVG(rating), 2) AS avg_rating,
    ROUND(AVG(recommended_ind) * 100, 1) AS recommend_rate_pct,
    COUNT(*) AS review_count
FROM reviews
GROUP BY department_name
ORDER BY avg_rating DESC
"""
rows1 = con.execute(q1).fetchall()
print("\nInterpretation: Tops and Dresses have the highest average ratings, "
      "while Trend has the lowest. Recommend rates closely track average "
      "ratings, suggesting customers are consistent in their sentiment.\n")
print(f"{'Department':<18} {'Avg Rating':<12} {'Recommend %':<14} {'Count':<8}")
print("-" * 52)
for r in rows1:
    print(f"{r[0]:<18} {r[1]:<12} {r[2]:<14} {r[3]:<8}")

# -----------------------------------------------------------------------
# Query 2: Average rating by age_bucket
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("QUERY 2: Average Rating by Age Bucket [Descriptive]")
print("=" * 70)

q2 = """
SELECT
    age_bucket,
    ROUND(AVG(rating), 2) AS avg_rating,
    COUNT(*) AS review_count
FROM reviews
GROUP BY age_bucket
ORDER BY
    CASE age_bucket
        WHEN '<25' THEN 1
        WHEN '25-34' THEN 2
        WHEN '35-44' THEN 3
        WHEN '45-54' THEN 4
        WHEN '55+' THEN 5
    END
"""
rows2 = con.execute(q2).fetchall()
print("\nInterpretation: Older reviewers (55+) give the highest average "
      "ratings, while the youngest (<25) are the most critical. Rating "
      "increases monotonically with age.\n")
print(f"{'Age Bucket':<15} {'Avg Rating':<12} {'Count':<8}")
print("-" * 35)
for r in rows2:
    print(f"{r[0]:<15} {r[1]:<12} {r[2]:<8}")

# -----------------------------------------------------------------------
# Query 3: Does review_length differ between recommended vs not?
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("QUERY 3: Review Length: Recommended vs Not Recommended [Diagnostic]")
print("=" * 70)

q3 = """
SELECT
    recommended_ind,
    ROUND(AVG(review_length), 1) AS avg_review_length,
    COUNT(*) AS review_count
FROM reviews
GROUP BY recommended_ind
ORDER BY recommended_ind DESC
"""
rows3 = con.execute(q3).fetchall()
print("\nInterpretation: Non-recommended reviews are on average longer "
      "than recommended ones. This suggests customers who are dissatisfied "
      "write more detailed reviews to explain their negative experience.\n")
print(f"{'Recommended':<15} {'Avg Word Count':<16} {'Count':<8}")
print("-" * 39)
for r in rows3:
    label = "Yes" if r[0] == 1 else "No"
    print(f"{label:<15} {r[1]:<16} {r[2]:<8}")

# Also compute statistical context: distribution range
q3b = """
SELECT
    recommended_ind,
    MIN(review_length) AS min_len,
    ROUND(AVG(review_length), 1) AS avg_len,
    MAX(review_length) AS max_len,
    ROUND(STDDEV_SAMP(review_length), 1) AS std_len,
    COUNT(*) AS cnt
FROM reviews
GROUP BY recommended_ind
ORDER BY recommended_ind DESC
"""
rows3b = con.execute(q3b).fetchall()
print("\nDetailed statistics:")
print(f"{'Rec':<5} {'Min':<6} {'Avg':<8} {'Max':<6} {'StdDev':<8} {'Count':<8}")
print("-" * 41)
for r in rows3b:
    label = "Yes" if r[0] == 1 else "No"
    print(f"{label:<5} {r[1]:<6} {r[2]:<8} {r[3]:<6} {r[4]:<8} {r[5]:<8}")

# -----------------------------------------------------------------------
# Query 4: Which Class Name has lowest recommend-rate & why?
# -----------------------------------------------------------------------
print("\n" + "=" * 70)
print("QUERY 4: Lowest Recommend Rate by Class Name [Diagnostic]")
print("=" * 70)

q4a = """
SELECT
    class_name,
    ROUND(AVG(recommended_ind) * 100, 1) AS recommend_rate_pct,
    ROUND(AVG(rating), 2) AS avg_rating,
    COUNT(*) AS review_count
FROM reviews
GROUP BY class_name
ORDER BY recommend_rate_pct ASC
LIMIT 5
"""
rows4a = con.execute(q4a).fetchall()
print("\nInterpretation: Chemises, Casual bottoms, and Intimates have the "
      "lowest recommend rates. However, Chemises and Casual bottoms have "
      "very few reviews (1-2), so the rate is unreliable. Among well-reviewed "
      "classes, Intimates (avg rating 3.62) and Lounge (avg rating 3.97) have "
      "the lowest recommend rates. The low recommend rate for Intimates is "
      "explained by a lower average rating — indicating product quality/fit "
      "issues rather than unrelated factors.\n")
print(f"{'Class Name':<20} {'Recommend %':<14} {'Avg Rating':<12} {'Count':<8}")
print("-" * 54)
for r in rows4a:
    print(f"{r[0]:<20} {r[1]:<14} {r[2]:<12} {r[3]:<8}")

# Dig deeper: for the bottom class with enough reviews, check rating distribution
print("\nRating distribution for the bottom 5 class_names:")
q4b = """
WITH bottom_classes AS (
    SELECT class_name
    FROM reviews
    GROUP BY class_name
    HAVING COUNT(*) >= 20
    ORDER BY AVG(recommended_ind) ASC
    LIMIT 5
)
SELECT
    r.class_name,
    r.rating,
    COUNT(*) AS cnt,
    ROUND(AVG(r.recommended_ind) * 100, 1) AS recommend_rate_pct
FROM reviews r
JOIN bottom_classes b ON r.class_name = b.class_name
GROUP BY r.class_name, r.rating
ORDER BY r.class_name, r.rating DESC
"""
rows4b = con.execute(q4b).fetchall()
print(f"{'Class Name':<20} {'Rating':<8} {'Count':<8} {'Recommend %':<14}")
print("-" * 50)
for r in rows4b:
    print(f"{r[0]:<20} {r[1]:<8} {r[2]:<8} {r[3]:<14}")

con.close()

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)