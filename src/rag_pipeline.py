from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from src.retriever import get_retriever
from src.llm import get_llm

def create_rag_chain(vectorstore):
    """Returns the vectorstore to be used by the pipeline."""
    return vectorstore

def answer_question(vectorstore, query: str, chat_history: List = None) -> Dict[str, Any]:
    """Answers a question using the LLM and vectorstore, and returns answer + source documents."""
    if chat_history is None:
        chat_history = []
        
    llm = get_llm()
    retriever = get_retriever(vectorstore, top_k=4, search_type="mmr")
    
    import time
    
    docs = retriever.invoke(query)
    print(f"[DEBUG] Retrieved {len(docs)} chunks for the query.")
    
    context = "\n\n".join([f"Document chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
    
    system_prompt = (
        "You are a professional AI Knowledge Assistant for answering questions about uploaded documents.\n"
        "Use the following pieces of retrieved context to answer the question.\n"
        "If you don't know the answer or the information is not in the context, just say exactly: "
        "'I couldn't find this information in the uploaded documents.'\n"
        "Do NOT hallucinate or fabricate information.\n"
        "Do NOT invent citations.\n"
        "Rely strictly on the provided context.\n\n"
        "FORMATTING INSTRUCTIONS:\n"
        "- Please format your response beautifully using markdown.\n"
        "- Use clear paragraphs for readability.\n"
        "- Use bullet points or numbered lists when listing features, steps, or multiple items.\n"
        "- Bold key terms to make the answer easy to scan and understand.\n\n"
        f"Context:\n{context}"
    )
    
    messages = [("system", system_prompt)]
    
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            messages.append(("human", msg.content))
        elif isinstance(msg, AIMessage):
            messages.append(("assistant", msg.content))
            
    messages.append(("human", query))
    
    prompt = ChatPromptTemplate.from_messages(messages)
    
    print(f"[DEBUG] Selected LLM Model: {getattr(llm, 'model', 'FallbackChain')}")
    print("[DEBUG] Sending grounded prompt to Gemini...")
    start_time = time.time()
    
    try:
        response = llm.invoke(prompt.format_messages())
        elapsed = time.time() - start_time
        print(f"[DEBUG] Successfully received Gemini response in {elapsed:.2f}s.")
        answer_text = response.content
        if isinstance(answer_text, list):
            answer_text = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in answer_text])
        elif not isinstance(answer_text, str):
            answer_text = str(answer_text)
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[DEBUG] Gemini API call failed after {elapsed:.2f}s with exception: {e}")
        raise e
    
    return {
        "answer": answer_text,
        "source_documents": docs
    }
