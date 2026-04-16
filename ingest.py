from pathlib import Path
from typing import List, Dict

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "math_docs"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def read_pdf(file_path: Path) -> str:
    """Read text from a PDF file."""
    reader = PdfReader(str(file_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)


def read_text_file(file_path: Path) -> str:
    """Read text from a plain text-like file."""
    return file_path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: Path) -> List[Dict]:
    """Load supported documents from the data directory."""
    docs = []

    for file_path in data_dir.iterdir():
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                text = read_pdf(file_path)
            elif suffix in {".txt", ".md", ".tex"}:
                text = read_text_file(file_path)
            else:
                print(f"Skipping unsupported file: {file_path.name}")
                continue

            if text.strip():
                docs.append(
                    {
                        "source": file_path.name,
                        "text": text,
                    }
                )
                print(f"Loaded: {file_path.name}")
            else:
                print(f"Empty text, skipping: {file_path.name}")

        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")

    return docs


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping character-based chunks."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - chunk_overlap

    return chunks


def build_chunk_records(docs: List[Dict]) -> List[Dict]:
    """Create chunk-level records with metadata."""
    records = []

    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{doc['source']}_chunk_{i}",
                    "text": chunk,
                    "metadata": {
                        "source": doc["source"],
                        "chunk_index": i,
                    },
                }
            )

    return records


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents.")

    if not docs:
        print("No valid documents found in data/. Exiting.")
        return

    print("Chunking documents...")
    records = build_chunk_records(docs)
    print(f"Created {len(records)} chunks.")

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [r["text"] for r in records]
    ids = [r["id"] for r in records]
    metadatas = [r["metadata"] for r in records]

    print("Generating embeddings...")
    embeddings = embed_model.encode(texts, show_progress_bar=True).tolist()

    print("Opening ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    print("Adding chunks to collection...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("Done!")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Stored {len(records)} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    main()