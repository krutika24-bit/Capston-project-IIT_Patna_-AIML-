"""
aggregate_findings.py

Joins extracted_reviews.csv back to original review data to get Department Name
and Class Name, then prints complaint_category counts by department/class to
identify which product lines merchandising should investigate.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_CSV = "data/raw/womens_clothing_reviews.csv"
EXTRACTED_CSV = "data/extracted_reviews.csv"

def main():
    # Load original data to get Department Name and Class Name
    raw_df = pd.read_csv(RAW_CSV)
    
    # Load extracted results
    extracted_df = pd.read_csv(EXTRACTED_CSV)
    
    # Join on index to get department/class info
    merged = raw_df.loc[extracted_df["index"]].copy()
    merged["sentiment"] = extracted_df["sentiment"].values
    merged["complaint_category"] = extracted_df["complaint_category"].values
    merged["would_recommend_reason"] = extracted_df["would_recommend_reason"].values
    
    # Aggregate complaint_category counts by department and class
    print("\n" + "=" * 60)
    print("COMPLAINT CATEGORY COUNTS BY DEPARTMENT")
    print("=" * 60)
    
    dept_complaints = merged.groupby(["Department Name", "complaint_category"]).size().unstack(fill_value=0)
    print(dept_complaints.to_string())
    
    print("\n" + "=" * 60)
    print("COMPLAINT CATEGORY COUNTS BY CLASS")
    print("=" * 60)
    
    class_complaints = merged.groupby(["Class Name", "complaint_category"]).size().unstack(fill_value=0)
    print(class_complaints.to_string())
    
    # Prescriptive recommendations
    print("\n" + "=" * 60)
    print("PRESCRIPTIVE RECOMMENDATIONS FOR MERCHANDISING")
    print("=" * 60)
    
    for dept in dept_complaints.index:
        row = dept_complaints.loc[dept]
        top_complaint = row.idxmax()
        top_count = row.max()
        if top_complaint != "none" and top_count > 0:
            print(f"\nWARNING: {dept}: Highest complaint category is '{top_complaint}' ({top_count} instances)")
            if top_complaint == "sizing":
                print("   -> Review and update the size chart for this department")
            elif top_complaint == "quality":
                print("   -> Investigate supplier quality standards")
            elif top_complaint == "fabric":
                print("   -> Consider alternative fabric sourcing options")
            elif top_complaint == "color_mismatch":
                print("   -> Review product photography/color representation")

if __name__ == "__main__":
    main()