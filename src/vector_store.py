import os
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from src.embeddings import get_embeddings_model

def get_vector_store() -> Chroma:
    """Gets an in-memory Chroma vector store."""
    embeddings = get_embeddings_model()
    return Chroma(embedding_function=embeddings)

def add_documents_to_vector_store(chunks: List[Document]) -> Chroma:
    """Adds document chunks to an in-memory Chroma vector store."""
    embeddings = get_embeddings_model()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    return vectorstore

def clear_vector_store():
    """No-op for in-memory store."""
    pass
