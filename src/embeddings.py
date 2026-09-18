from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings_model():
    """Returns the Sentence Transformers embedding model."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
