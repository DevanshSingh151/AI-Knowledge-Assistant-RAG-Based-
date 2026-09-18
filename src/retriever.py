from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

def get_retriever(vectorstore: Chroma, top_k: int = 4, search_type: str = "mmr") -> VectorStoreRetriever:
    """Returns a retriever for the given vector store."""
    if search_type == "mmr":
        return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": top_k, "fetch_k": top_k * 3})
    else:
        return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
