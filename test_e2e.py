import os
import time
from reportlab.pdfgen import canvas
from src.document_loader import load_pdfs
from src.text_processor import process_documents
from src.vector_store import add_documents_to_vector_store, clear_vector_store
from src.rag_pipeline import create_rag_chain, answer_question
from langchain_core.messages import HumanMessage, AIMessage

def create_dummy_pdf(filename, text_lines):
    c = canvas.Canvas(filename)
    y = 750
    for line in text_lines:
        c.drawString(100, y, line)
        y -= 20
    c.save()

def main():
    os.makedirs("data/temp_uploads", exist_ok=True)
    create_dummy_pdf("data/temp_uploads/doc1.pdf", [
        "The main objective of the project is to build an AI Knowledge Assistant.",
        "It uses a local ChromaDB for vector storage.",
        "The retrieval algorithm used is Maximal Marginal Relevance (MMR)."
    ])
    create_dummy_pdf("data/temp_uploads/doc2.pdf", [
        "Optional features include semantic search and source citations.",
        "These features impact performance slightly but improve accuracy.",
        "Based on all documents, the conclusion is that RAG is very effective."
    ])

    print("1. Testing PDF Loader...")
    docs = load_pdfs(["data/temp_uploads/doc1.pdf", "data/temp_uploads/doc2.pdf"])
    print(f"Loaded {len(docs)} documents.")

    print("2. Testing Chunking...")
    chunks = process_documents(docs, chunk_size=800, chunk_overlap=120)
    print(f"Created {len(chunks)} chunks.")

    print("3. Testing Embeddings & Vector DB...")
    clear_vector_store()
    vs = add_documents_to_vector_store(chunks)

    print("4. Testing RAG Pipeline & Gemini Generation...")
    rag_chain = create_rag_chain(vs)

    print("\n--- Test Direct ---")
    res = answer_question(rag_chain, "What is the main objective of the project?")
    print("A:", res["answer"])
    print("Sources:", [d.metadata.get('source') for d in res["source_documents"]])

    print("\n--- Test Unsupported ---")
    res = answer_question(rag_chain, "What is the capital of France?")
    print("A:", res["answer"])

    print("\n--- Test Conversation History ---")
    history = [
        HumanMessage(content="What are the optional features?"), 
        AIMessage(content="Semantic search and source citations.")
    ]
    res = answer_question(rag_chain, "How do they impact performance?", chat_history=history)
    print("A:", res["answer"])
    
    print("\nRunning Evaluation Script...")
    from evaluation.evaluate import run_evaluation
    run_evaluation()

if __name__ == "__main__":
    main()
