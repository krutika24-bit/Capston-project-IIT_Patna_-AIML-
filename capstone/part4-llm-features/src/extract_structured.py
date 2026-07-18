"""
extract_structured.py

Reads raw reviews CSV, samples 300 rows with non-null Review Text,
calls the Anthropic API (claude-sonnet-4-6) to extract structured
sentiment / complaint / recommendation data per review, and saves results.
"""

import os
import json
import time
import sys

import pandas as pd
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_CSV = "data/raw/womens_clothing_reviews.csv"
OUTPUT_CSV = "data/extracted_reviews.csv"
SAMPLE_SIZE = 300
SLEEP_SECONDS = 1.0  # rate-limit guard
MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Step 1 – load, filter, sample
# ---------------------------------------------------------------------------
def prepare_data():
    df = pd.read_csv(RAW_CSV)
    # Keep only rows where Review Text is not null
    df = df[df["Review Text"].notna()].reset_index(drop=True)
    if len(df) < SAMPLE_SIZE:
        print(
            f"WARNING: only {len(df)} non-null reviews available, "
            f"using all of them.",
            file=sys.stderr,
        )
        sampled = df
    else:
        sampled = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    print(f"Loaded {len(sampled)} reviews -> processing")
    return sampled


# ---------------------------------------------------------------------------
# Step 2 – build the prompt (with one few-shot example)
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLE = (
    'Review: "I love this dress! The fabric is beautiful and fits perfectly."\n'
    '{"sentiment": "positive", "complaint_category": "none", '
    '"would_recommend_reason": "beautiful fabric and perfect fit"}'
)

SYSTEM_PROMPT = (
    "You are a helpful assistant that extracts structured data from product "
    "reviews. Return ONLY valid JSON (no markdown fences, no extra text). "
    "The JSON must have exactly these three keys:\n"
    '  "sentiment": "positive", "negative", or "neutral"\n'
    '  "complaint_category": one of "sizing", "quality", "fabric", '
    '"color_mismatch", "none", "other"\n'
    '  "would_recommend_reason": a short string explaining why the reviewer '
    "would or would not recommend the product\n\n"
    f"Example:\n{FEW_SHOT_EXAMPLE}"
)

USER_PROMPT_TEMPLATE = 'Review: "{text}"'


def build_prompt(review_text: str):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=review_text)},
    ]


# ---------------------------------------------------------------------------
# Step 3 – safe API call with one retry
# ---------------------------------------------------------------------------
def extract_from_review(client: Anthropic, review_text: str) -> dict:
    messages = build_prompt(review_text)

    for attempt in range(2):  # first attempt + one retry
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=256,
                temperature=0.0,
                messages=messages,
            )
            raw = resp.content[0].text.strip()
            # If the model wraps it in markdown code fences, strip them
            if raw.startswith("```"):
                lines = raw.splitlines()
                # Remove first and last line if they are ``` markers
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            parsed = json.loads(raw)

            # Validate required keys
            for key in ("sentiment", "complaint_category", "would_recommend_reason"):
                if key not in parsed:
                    raise ValueError(f"Missing key: {key}")
            return parsed

        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            if attempt == 0:
                print(f"  Retry on parse error: {exc}", file=sys.stderr)
                time.sleep(SLEEP_SECONDS)
                continue
            # Fallback on total failure
            return {
                "sentiment": "neutral",
                "complaint_category": "none",
                "would_recommend_reason": "parse-failed",
            }


# ---------------------------------------------------------------------------
# Step 4 – main orchestration
# ---------------------------------------------------------------------------
def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = Anthropic(api_key=api_key)

    # Prepare data
    df = prepare_data()

    records = []
    total = len(df)

    for idx, row in df.iterrows():
        review_text = str(row["Review Text"])
        print(f"[{idx + 1}/{total}] Processing review ...", file=sys.stderr)

        result = extract_from_review(client, review_text)

        records.append(
            {
                "index": idx,
                "sentiment": result["sentiment"],
                "complaint_category": result["complaint_category"],
                "would_recommend_reason": result["would_recommend_reason"],
            }
        )

        time.sleep(SLEEP_SECONDS)

    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone – {len(out_df)} results written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()