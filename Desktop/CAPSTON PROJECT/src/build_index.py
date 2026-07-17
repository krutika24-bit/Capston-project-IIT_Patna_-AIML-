"""
build_index.py

Chunks Review Text (splits only if over ~150 words), embeds them using
sentence-transformers (all-MiniLM-L6-v2), and stores vectors + metadata
(department, class, rating, age) in a local ChromaDB collection at data/chroma/.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RAW_CSV = "data/raw/womens_clothing_reviews.csv"
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "reviews"
CHUNK_WORD_THRESHOLD = 150
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text: str, max_words: int = CHUNK_WORD_THRESHOLD) -> list:
    """Split text into chunks if it exceeds max_words. Returns list of chunks."""
    words = text.split()
    if len(words) <= max_words:
        return [text]
    
    # Split into multiple chunks
    chunks = []
    current_chunk = []
    
    for i, word in enumerate(words):
        current_chunk.append(word)
        if len(current_chunk) >= max_words and i < len(words) - 1:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def main():
    # Load reviews
    df = pd.read_csv(RAW_CSV)
    df = df[df["Review Text"].notna()].reset_index(drop=True)
    
    print(f"Loaded {len(df)} reviews to index")
    
    # Initialize ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Load embedding model
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Process and embed reviews
    ids = []
    embeddings = []
    metadata = []
    documents = []
    
    for idx, row in df.iterrows():
        review_text = str(row["Review Text"])
        chunks = chunk_text(review_text)
        
        for chunk_idx, chunk in enumerate(chunks):
            doc_id = f"{idx}_{chunk_idx}" if len(chunks) > 1 else str(idx)
            ids.append(doc_id)
            documents.append(chunk)
            embeddings.append(model.encode(chunk).tolist())
            
            metadata.append({
                "department": row.get("Department Name", "Unknown"),
                "class": row.get("Class Name", "Unknown"),
                "rating": int(row.get("Rating", 0)),
                "age": int(row.get("Age", 0)),
                "original_index": idx,
            })
    
    # Store in ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadata,
    )
    
    print(f"\nIndexed {len(ids)} chunks into ChromaDB at {CHROMA_PATH}")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()