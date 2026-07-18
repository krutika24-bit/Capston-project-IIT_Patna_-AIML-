"""
app.py - Streamlit Dashboard for LLM Review Analysis

Run with: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import os

# Optional imports - handled gracefully if not installed
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_CSV = "data/raw/womens_clothing_reviews.csv"
EXTRACTED_CSV = "data/extracted_reviews.csv"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "reviews"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return None
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource
def get_chroma_collection():
    if not CHROMADB_AVAILABLE:
        st.warning("chromadb not installed. Install with: pip install chromadb")
        return None
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        return client.get_or_create_collection(name=COLLECTION_NAME)
    except Exception as e:
        st.error(f"Could not connect to ChromaDB: {e}")
        return None

@st.cache_data
def load_data():
    raw_df = pd.read_csv(RAW_CSV)
    try:
        extracted_df = pd.read_csv(EXTRACTED_CSV)
        return raw_df, extracted_df
    except FileNotFoundError:
        return raw_df, None

def search_reviews(question, model, collection, k=5):
    if collection is None:
        return [], [], []
    
    query_embedding = model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
    )
    
    documents = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    return documents, ids, metadatas

# ---------------------------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="LLM Review Analysis", layout="wide")
st.title("LLM Review Analysis Dashboard")

# Load data
raw_df, extracted_df = load_data()
model = load_embedding_model()
collection = get_chroma_collection()

# Tabs
tab1, tab2, tab3 = st.tabs(["Complaint Analysis", "Ask Questions", "Review Explorer"])

with tab1:
    st.header("Complaint Analysis by Department")
    
    if extracted_df is not None:
        # Merge extracted with raw data
        merged = raw_df.loc[extracted_df["index"]].copy()
        merged["sentiment"] = extracted_df["sentiment"].values
        merged["complaint_category"] = extracted_df["complaint_category"].values
        
        # Show complaint counts
        dept_complaints = merged.groupby(["Department Name", "complaint_category"]).size().unstack(fill_value=0)
        st.dataframe(dept_complaints.style.background_gradient(cmap="RdYlGn_r", axis=1))
        
        # Show prescriptive recommendations
        st.subheader("Prescriptive Recommendations")
        for dept in dept_complaints.index:
            row = dept_complaints.loc[dept]
            top_complaint = row.idxmax()
            top_count = row.max()
            if top_complaint != "none" and top_count > 0:
                if top_complaint == "sizing":
                    st.warning(f"**{dept}**: Review size chart - {top_count} sizing complaints")
                elif top_complaint == "quality":
                    st.warning(f"**{dept}**: Investigate supplier quality - {top_count} quality complaints")
                elif top_complaint == "fabric":
                    st.warning(f"**{dept}**: Consider fabric sourcing - {top_count} fabric complaints")
                else:
                    st.info(f"**{dept}**: {top_count} {top_complaint} complaints")
    else:
        st.info("Run `python src/extract_structured.py` to generate extracted data")

with tab2:
    st.header("Ask Questions (RAG)")
    
    question = st.text_input("Enter your question about the reviews:")
    
    if st.button("Ask") and question:
        if os.environ.get("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            
            documents, ids, metadatas = search_reviews(question, model, collection)
            
            if documents:
                st.subheader("Top 5 Similar Reviews")
                for i, (doc, doc_id, meta) in enumerate(zip(documents, ids, metadatas)):
                    with st.expander(f"Review {doc_id} - {meta.get('department', 'Unknown')}"):
                        st.write(doc)
                        st.caption(f"Rating: {meta.get('rating', 'Unknown')} | Age: {meta.get('age', 'Unknown')}")
                
                # Build context
                context_lines = []
                for doc, doc_id, meta in zip(documents, ids, metadatas):
                    context_lines.append(
                        f"[Review {doc_id}] Department: {meta.get('department', 'Unknown')}, "
                        f"Class: {meta.get('class', 'Unknown')}, Rating: {meta.get('rating', 'Unknown')}\n{doc}"
                    )
                context = "\n\n".join(context_lines)
                
                # Call API
                try:
                    client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
                    system_prompt = (
                        "You are a helpful assistant that answers questions using ONLY the provided context. "
                        "If you cannot answer the question from the context, respond with exactly: "
                        "\"not found in context\". Do not use any external knowledge. "
                        "When you answer, cite the review IDs you used."
                    )
                    
                    with st.spinner("Thinking..."):
                        resp = client.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=512,
                            temperature=0.0,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                            ],
                        )
                    st.subheader("Answer")
                    st.write(resp.content[0].text)
                except Exception as e:
                    st.error(f"API Error: {e}")
            else:
                st.info("No relevant reviews found. Run `src/build_index.py` first.")
        else:
            st.error("ANTHROPIC_API_KEY not set. Set it as an environment variable.")

with tab3:
    st.header("Review Explorer")
    
    if extracted_df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            dept_filter = st.selectbox("Department", ["All"] + list(raw_df["Department Name"].unique()))
        
        with col2:
            sentiment_filter = st.selectbox("Sentiment", ["All", "positive", "negative", "neutral"])
        
        merged = raw_df.loc[extracted_df["index"]].copy()
        merged["sentiment"] = extracted_df["sentiment"].values
        merged["complaint_category"] = extracted_df["complaint_category"].values
        
        if dept_filter != "All":
            merged = merged[merged["Department Name"] == dept_filter]
        if sentiment_filter != "All":
            merged = merged[merged["sentiment"] == sentiment_filter]
        
        st.write(f"Showing {len(merged)} reviews")
        
        for _, row in merged.head(50).iterrows():
            with st.expander(f"Rating: {row['Rating']} - {row.get('Department Name', 'Unknown')}"):
                st.write(f"**Review Text:** {row['Review Text']}")
                st.caption(f"Age: {row.get('Age', 'Unknown')}")
    else:
        st.info("Run `python src/extract_structured.py` to generate extracted data")