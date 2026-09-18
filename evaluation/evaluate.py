import json
import os
import sys

# Add parent directory to path so we can import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import create_rag_chain, answer_question
from src.vector_store import get_vector_store

def run_evaluation():
    questions_path = os.path.join(os.path.dirname(__file__), "questions.json")
    with open(questions_path, "r") as f:
        questions = json.load(f)
    
    print("Starting Evaluation...")
    try:
        vs = get_vector_store()
        rag_chain = create_rag_chain(vs)
    except Exception as e:
        print("Could not initialize vector store or RAG chain. Please ensure documents are processed first.")
        print(f"Error: {e}")
        return

    results = []
    
    for q in questions:
        print(f"\n[{q['type']}] Question: {q['question']}")
        try:
            result = answer_question(rag_chain, q["question"])
            answer = result["answer"]
            sources = result["source_documents"]
            
            print(f"Answer: {answer}")
            
            # Simple check if sources were returned
            if sources:
                print(f"Sources: {[doc.metadata.get('source', 'Unknown') for doc in sources]}")
            else:
                print("Sources: None")
                
            results.append({
                "id": q["id"],
                "question": q["question"],
                "answer": answer,
                "sources_count": len(sources)
            })
            
        except Exception as e:
            print(f"Error evaluating question {q['id']}: {e}")

    print("\nEvaluation Complete.")
    print(f"Evaluated {len(results)} out of {len(questions)} questions.")

if __name__ == "__main__":
    run_evaluation()
