import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Loads text from multiple PDFs."""
    all_documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = os.path.basename(path)
            # PyPDFLoader usually stores page in doc.metadata['page']
        all_documents.extend(docs)
    return all_documents
