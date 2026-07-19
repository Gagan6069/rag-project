import json
import os

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter   
from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.startup_validator import StartupValidator

from src.config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    CHUNK_STORE_PATH,
)

StartupValidator.validate_for_indexing()


def build_vector_db():

    print("Loading PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    docs = loader.load()

    print(f"Loaded pages: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    # Add stable chunk IDs to metadata
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    embedding = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    db = FAISS.from_documents(chunks, embedding)

    db.save_local(str(FAISS_INDEX_PATH))

    print(f"FAISS index saved at: {FAISS_INDEX_PATH}")

    # Save chunks for BM25
    # os.makedirs(os.path.dirname(CHUNK_STORE_PATH), exist_ok=True)

    chunk_records = []

    for chunk in chunks:

        chunk_records.append({
            "chunk_id": chunk.metadata.get("chunk_id"),
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
        })

    with open(CHUNK_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_records, f, ensure_ascii=False, indent=2)

    print(f"Chunk store saved at: {CHUNK_STORE_PATH}")


if __name__ == "__main__":
    build_vector_db()