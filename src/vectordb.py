from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    PDF_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
)

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

chunks = splitter.split_documents(docs)

embedding = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

db = FAISS.from_documents(chunks, embedding)

db.save_local(FAISS_INDEX_PATH)

print("Vector database created successfully.")