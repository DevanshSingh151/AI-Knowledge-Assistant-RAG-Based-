import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings_model

def get_vector_store(persist_directory: str = "data/chroma_db") -> Chroma:
    """Gets or creates a Chroma vector store."""
    embeddings = get_embeddings_model()
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def add_documents_to_vector_store(chunks: List[Document], persist_directory: str = "data/chroma_db") -> Chroma:
    """Adds document chunks to the Chroma vector store."""
    embeddings = get_embeddings_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def clear_vector_store(persist_directory: str = "data/chroma_db"):
    """Clears the existing vector store data (if any)."""
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory, ignore_errors=True)
