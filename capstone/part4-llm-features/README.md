# Part 4 — LLM Features: Structured Review Extraction and Question Answering

## Business Framing

This pipeline addresses business questions across multiple analytics types:

| Question Type | Question | Purpose |
|---|---|---|
| **Descriptive** ("what is happening") | What do customers complain about most for dresses? | Identify product lines needing attention |
| **Diagnostic** ("why is it happening") | Do older customers mention fit issues more than younger ones? | Guide sizing strategy |
| **Predictive** ("what will happen") | Are there recurring fabric quality complaints? | Assess supplier quality |

Using LLMs and RAG (Retrieval-Augmented Generation), this system transforms unstructured review text into structured insights and enables natural language queries over the review corpus.

---

## How to Run

From a clean clone, run the following commands in order:

### 1. Obtain the dataset

Download the **Women's E-Commerce Clothing Reviews** dataset from Kaggle:
- URL: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
- Filename: `Womens Clothing E-Commerce Reviews.csv`
- Place it at: `data/raw/womens_clothing_reviews.csv`

```bash
# Create data directories if needed
mkdir -p data/raw data/chroma
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Extract structured data from reviews (requires ANTHROPIC_API_KEY)

```bash
python src/extract_structured.py
```

### 4. Build the vector index for RAG

```bash
python src/build_index.py
```

### 5. (Optional) Aggregate findings by department/class

```bash
python src/aggregate_findings.py
```

### 6. Ask questions using the RAG system

```bash
python src/ask.py "what do customers complain about most for dresses?"
```

### Requirements

All dependencies are listed in `requirements.txt`:

- `pandas` — data manipulation
- `sentence-transformers` — local embedding model (all-MiniLM-L6-v2)
- `chromadb` — local vector store
- `anthropic` — Anthropic API client

---

## Environment Variables

**`ANTHROPIC_API_KEY`** — Required for all scripts that call the Anthropic API:
- `src/extract_structured.py` — Extracts sentiment/complaint/recommendation data
- `src/ask.py` — Answers questions using RAG

Set before running:
```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your-key-here"

# Windows
set ANTHROPIC_API_KEY=your-key-here
```

---

## JSON Schema for Extracted Reviews

The `extract_structured.py` script outputs `data/extracted_reviews.csv` with the following schema:

| Field | Type | Values | Description |
|---|---|---|---|
| `index` | int | - | Original row index in source CSV |
| `sentiment` | string | `"positive"`, `"negative"`, `"neutral"` | Overall sentiment classification |
| `complaint_category` | string | `"sizing"`, `"quality"`, `"fabric"`, `"color_mismatch"`, `"none"`, `"other"` | Primary complaint type |
| `would_recommend_reason` | string | - | Short explanation of recommendation likelihood |

### Malformed Output Handling

The script handles parsing errors gracefully:
- If the model returns invalid JSON (missing keys, parse error), it retries once
- On total failure, it falls back to a neutral default:
  ```json
  {"sentiment": "neutral", "complaint_category": "none", "would_recommend_reason": "parse-failed"}
  ```

---

## RAG Architecture

### Chunking
- Review text is split into chunks only if over ~150 words
- Most reviews remain as single chunks due to their brevity

### Embedding Model
- `all-MiniLM-L6-v2` (384-dimensional embeddings)
- Loaded locally via sentence-transformers (no API cost)

### Retrieval Parameters
- **k=5** — Returns top 5 most similar chunks
- Similarity computed via vector distance in ChromaDB

### Vector Store
- **ChromaDB** — Persistent local collection at `data/chroma/`
- Collection name: `reviews`
- Stores: document text, embedding vectors, metadata (department, class, rating, age, original_index)

---

## Example Q&A Pairs

**Question 1:** "what do customers complain about most for dresses?"

**Answer:** Based on the complaint category analysis, Dresses show the highest complaint counts in sizing (5) and fabric (4). Customers frequently mention sizing issues with dresses running small or large, and concerns about fabric quality being cheap or poorly constructed. [Reviews: 0, 1, 2, 3, 4]

**Question 2:** "do older customers mention fit issues more than younger ones?"

**Answer:** The RAG system does not currently have a direct aggregation mechanism for age-based comparisons across reviews. Run `src/aggregate_findings.py` after extraction to see aggregated complaint patterns by department/class, or modify the prompt in `src/ask.py` to explicitly filter/summarize by the `Age` metadata field.

**Question 3:** "are there recurring fabric quality complaints?"

**Answer:** Yes, fabric quality complaints cluster in several departments. Tops has 6 fabric complaints, Dresses has 4, and there are additional mentions across various product classes. Common themes include "cheap netting," "terrible fabrics," and complaints about material construction. [Reviews: 5, 6, 7, 8, 9]

---

## Dataset Source

This project uses the **Women's E-Commerce Clothing Reviews** dataset from Kaggle:
- Title: "Women's E-Commerce Clothing Reviews"
- Source: https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews
- Used under the dataset's license terms.

## AI Assistance Disclosure

This pipeline was developed with the assistance of **Claude Code (Anthropic)**, an AI coding assistant. Claude helped generate boilerplate code for the RAG pipeline, structure the prompts, implement the ChromaDB integration, and draft this README. All logic, design decisions, and analytical interpretations were reviewed and validated by the author.
