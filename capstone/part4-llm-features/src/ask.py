"""
ask.py

CLI tool: python src/ask.py "question" retrieves top-5 similar review chunks from
ChromaDB, then calls Anthropic API with those chunks as context, instructed to
answer ONLY from the provided context and cite which reviews it used (by index/id),
or say "not found in context" if it can't answer.
"""

import sys
import os
from sentence_transformers import SentenceTransformer
import chromadb
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHROMA_PATH = "data/chroma"
COLLECTION_NAME = "reviews"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MODEL = "claude-sonnet-4-6"
TOP_K = 5


def get_similar_chunks(question: str, model) -> list:
    """Retrieve top-5 similar review chunks from ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Embed the query
    query_embedding = model.encode(question).tolist()
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
    )
    
    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/ask.py \"question\"", file=sys.stderr)
        sys.exit(1)
    
    question = sys.argv[1]
    
    # Check for API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.", file=sys.stderr)
        sys.exit(1)
    
    # Load embedding model
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Get similar chunks
    results = get_similar_chunks(question, embed_model)
    
    documents = results.get("documents", [[]])[0]
    ids = results.get("ids", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    
    if not documents:
        print("No relevant reviews found in the index. Run src/build_index.py first.")
        return
    
    # Build context
    context_lines = []
    for i, (doc, doc_id, meta) in enumerate(zip(documents, ids, metadatas)):
        context_lines.append(
            f"[Review {doc_id}] Department: {meta.get('department', 'Unknown')}, "
            f"Class: {meta.get('class', 'Unknown')}, Rating: {meta.get('rating', 'Unknown')}, "
            f"Age: {meta.get('age', 'Unknown')}\n{doc}"
        )
    
    context = "\n\n".join(context_lines)
    
    # Call Anthropic API
    client = Anthropic(api_key=api_key)
    
    system_prompt = (
        "You are a helpful assistant that answers questions using ONLY the provided context. "
        "If you cannot answer the question from the context, respond with exactly: "
        "\"not found in context\". Do not use any external knowledge. "
        "When you answer, cite the review IDs you used in your response."
    )
    
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    
    answer = resp.content[0].text.strip()
    print(answer)


if __name__ == "__main__":
    main()