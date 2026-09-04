# Capstone Project: Olist Delivery Analytics & Predictive Modeling
## IIT Patna — AI/ML Program

🚀 **A complete end-to-end data science project** combining diagnostic analysis, machine learning prediction, interactive dashboards, and LLM-powered insights.

---

## 📋 Project Overview

This capstone project analyzes **Olist e-commerce delivery data** to understand why deliveries are late, predict which orders will be delayed, and provide actionable insights to operations teams.

### Four Interconnected Parts:

| Part | Focus | Output |
|------|-------|--------|
| **Part 1** | 📊 **Diagnostic Analysis** — Why do deliveries fail? | CSV reports + statistical findings |
| **Part 2** | 🤖 **Predictive Model** — Which orders will be late? | ML model (Logistic Regression, 72.7% ROC-AUC) |
| **Part 3** | 📈 **Interactive Dashboard** — Explore & explore patterns | Streamlit web app (live deployment) |
| **Part 4** | 🧠 **LLM-Powered Insights** — Extract themes from reviews | RAG system + structured JSON |

---

## 🎯 Business Impact

- **Late deliveries cause a 1.54-point drop in customer reviews** (4.20 → 2.66 stars)
- **~52% of delivered orders arrive late** — a major operational challenge
- By predicting high-risk orders at purchase time, operations can intervene proactively

---

## 📁 Project Structure

```
Capston-project-IIT_Patna_-AIML-/
├── README.md                          # This file — project overview
├── requirements.txt                   # Dependencies for all parts
├── capstone/
│   ├── part1-diagnostic-analysis/
│   │   ├── README.md
│   │   ├── src/                       # Analysis scripts
│   │   └── data/                      # CSVs and DuckDB
│   │
│   ├── part2-predictive-model/
│   │   ├── README.md
│   │   ├── src/                       # Training, evaluation, prediction
│   │   ├── data/                      # model_dataset.csv
│   │   └── model/                     # Saved model.pkl + metadata
│   │
│   ├── part3-dashboard/
│   │   ├── README.md
│   │   ├── src/app.py                 # Streamlit app
│   │   └── model/                     # Model for inference
│   │
│   └── part4-llm-features/
│       ├── README.md
│       ├── src/                       # Extraction, indexing, RAG
│       └── data/                      # Reviews + vector store
│
└── .gitignore
```

---

## 🚀 Quick Start

### Option 1: Use the Live Dashboard (No Setup)

**Part 3 Dashboard is deployed online:**  
👉 **[Live Streamlit App](https://capston-project-iit-patna-aiml.streamlit.app)**

No installation needed — just visit and explore!

---

### Option 2: Run Locally

#### Prerequisites
- Python 3.8+
- Git

#### Step 1: Clone & Navigate
```bash
git clone https://github.com/krutika24-bit/Capston-project-IIT_Patna_-AIML-.git
cd Capston-project-IIT_Patna_-AIML-
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Choose Your Path

**Run the Dashboard:**
```bash
cd capstone/part3-dashboard
streamlit run src/app.py
```

**Run Predictive Model:**
```bash
cd capstone/part2-predictive-model
python src/02_train_eval.py          # Train & evaluate
python src/predict.py                # Score a single order
```

**Run LLM Feature Extraction** (requires Anthropic API key):
```bash
export ANTHROPIC_API_KEY="your-key-here"
cd capstone/part4-llm-features
python src/extract_structured.py     # Extract sentiment/complaints
python src/ask.py "your question"    # Query via RAG
```

---

## 📚 Detailed Documentation

Each part has its own **README with full setup instructions, methodology, and results:**

1. **[Part 1: Diagnostic Analysis](capstone/part1-diagnostic-analysis/README.md)**  
   *Why are deliveries late? Statistical deep dive.*

2. **[Part 2: Predictive Model](capstone/part2-predictive-model/README.md)**  
   *Can we predict late deliveries? ML pipeline + model comparison.*

3. **[Part 3: Dashboard](capstone/part3-dashboard/README.md)**  
   *Interactive analytics for operations teams. [Live app](https://capston-project-iit-patna-aiml.streamlit.app)*

4. **[Part 4: LLM Features](capstone/part4-llm-features/README.md)**  
   *Extract complaint themes & answer questions via RAG.*

---

## 🏆 Key Results

### Part 2 Model Performance
| Model | ROC-AUC | F1 | Precision | Recall |
|-------|---------|-----|-----------|--------|
| **Logistic Regression** ⭐ | **0.7273** | 0.6774 | 0.66 | 0.70 |
| Random Forest | 0.7215 | 0.6760 | 0.66 | 0.69 |

**Interpretation:** The model correctly identifies late orders 73% of the time across all probability thresholds, capturing 70% of actual late deliveries.

### Top Features Driving Delays
1. **`promised_days`** (49.8%) — Longer delivery promises = higher risk
2. **`freight_to_price_ratio`** (6.8%) — Remote/bulky shipments delayed
3. **`seller_hist_late_rate`** (6.3%) — Seller's track record matters
4. **`total_price`** (6.1%) — Order value correlates with handling
5. **`purchase_month`** (3.4%) — Seasonal effects (holiday rushes)

---

## 🛠️ Tech Stack

- **Data Processing:** Pandas, NumPy, DuckDB
- **ML:** scikit-learn, joblib
- **Visualization:** Plotly, Streamlit, Matplotlib
- **LLM/RAG:** Anthropic Claude API, ChromaDB, sentence-transformers
- **Deployment:** Streamlit Community Cloud

---

## 📊 Data

**Primary Dataset:** Olist Brazilian E-Commerce Dataset  
- ~5,000 orders with delivery times, customer reviews, and logistics info
- Dates: 2016–2018
- Fully anonymized

**Secondary Dataset (Part 4):** Women's E-Commerce Clothing Reviews (Kaggle)  
- Used for LLM feature extraction and RAG demonstrations

---

## 👨‍💻 Author

**Krutika** — IIT Patna, AI/ML Program  
[GitHub Profile](https://github.com/krutika24-bit)

---

## 🤝 AI Assistance

This project was developed with the assistance of **Claude Code (Anthropic)**, an AI coding assistant. Claude helped with:
- Boilerplate code generation and project structure
- ML pipeline design and sklearn workflows
- RAG system architecture
- Streamlit dashboard components
- Prompt engineering for LLM feature extraction

---

## 📄 License

This project is open source. See individual part READMEs for dataset attribution and license details.

---

## 📞 Questions or Feedback?

Feel free to:
- 🐛 [Open an issue](https://github.com/krutika24-bit/Capston-project-IIT_Patna_-AIML-/issues)
- 💬 Start a [discussion](https://github.com/krutika24-bit/Capston-project-IIT_Patna_-AIML-/discussions)
- 📧 Reach out directly

---

**Last Updated:** September 2026  
**Status:** ✅ Complete — All parts functional and deployable
