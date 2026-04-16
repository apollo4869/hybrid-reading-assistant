from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer


CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "math_docs"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return collection


def embed_query(query: str) -> List[float]:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embedding = model.encode(query).tolist()
    return embedding


def retrieve(query: str, top_k: int = 5) -> Dict:
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    return results


def print_results(results: Dict):
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No results found.")
        return

    print("\n=== Retrieval Results ===\n")

    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), start=1):
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "unknown")

        print(f"[Result {i}]")
        print(f"Source: {source}")
        print(f"Chunk index: {chunk_index}")
        print(f"Distance: {dist:.4f}")
        print("Snippet:")
        print(doc[:500])
        print("-" * 80)


if __name__ == "__main__":
    query = input("Enter your question: ").strip()
    results = retrieve(query=query, top_k=5)
    print_results(results)