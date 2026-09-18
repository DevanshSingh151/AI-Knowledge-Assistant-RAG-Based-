import os
import streamlit as st
from typing import List
from langchain_core.messages import HumanMessage, AIMessage

from src.document_loader import load_pdfs
from src.text_processor import process_documents
from src.vector_store import add_documents_to_vector_store, get_vector_store, clear_vector_store
from src.rag_pipeline import create_rag_chain, answer_question

st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for black-and-white theme
st.markdown("""
<style>
    .stApp {
        background-color: #000000;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0f0f0f;
        border-right: 1px solid #333333;
    }
    
    [data-testid="stHeader"] {
        background-color: #000000;
    }

    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
    .stButton>button {
        background-color: #ffffff;
        color: #000000;
        border: 1px solid #ffffff;
        border-radius: 4px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #e0e0e0;
        color: #000000;
        border-color: #e0e0e0;
    }

    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
        border-radius: 4px;
    }
    .streamlit-expanderContent {
        border: 1px solid #333333;
        border-top: none;
        background-color: #0a0a0a;
    }

    [data-testid="stFileUploader"] {
        background-color: #1a1a1a;
        border: 1px dashed #555555;
        border-radius: 4px;
        padding: 10px;
    }

    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #ffffff;
    }
    
    [data-testid="stChatMessage"] {
        background-color: #111111;
        border: 1px solid #333333;
        border-radius: 6px;
        padding: 10px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "stats" not in st.session_state:
        st.session_state.stats = {"docs": 0, "pages": 0, "chunks": 0}
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None

def process_uploaded_files(uploaded_files):
    if not uploaded_files:
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Save uploaded files temporarily
    temp_dir = "data/temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    
    pdf_paths = []
    for file in uploaded_files:
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        pdf_paths.append(path)
    
    status_text.text("Processing documents...")
    
    try:
        if st.session_state.vectorstore is not None:
            st.session_state.vectorstore = None
        import gc
        gc.collect()
        
        clear_vector_store()
        
        status_text.text("✓ Extracting text...")
        docs = load_pdfs(pdf_paths)
        progress_bar.progress(25)
        
        status_text.text("✓ Creating chunks...")
        chunks = process_documents(docs, chunk_size=800, chunk_overlap=120)
        progress_bar.progress(50)
        
        status_text.text("✓ Generating embeddings & Updating vector database...")
        vs = add_documents_to_vector_store(chunks)
        progress_bar.progress(100)
        
        st.session_state.vectorstore = vs
        st.session_state.rag_chain = create_rag_chain(vs)
        st.session_state.stats = {
            "docs": len(uploaded_files),
            "pages": len(docs),
            "chunks": len(chunks)
        }
        
        status_text.text("Ready!")
        st.session_state.chat_history = []
    except Exception as e:
        st.error(f"Error processing files: {e}")
    finally:
        for path in pdf_paths:
            try:
                os.remove(path)
            except:
                pass

def main():
    initialize_session_state()
    
    with st.sidebar:
        st.markdown("### KNOWLEDGE BASE")
        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        if st.button("Process Documents"):
            process_uploaded_files(uploaded_files)
        
        st.markdown("---")
        st.markdown("### STATISTICS")
        st.write(f"Documents: {st.session_state.stats['docs']}")
        st.write(f"Pages: {st.session_state.stats['pages']}")
        st.write(f"Chunks: {st.session_state.stats['chunks']}")
        
        st.markdown("---")
        st.markdown("### RETRIEVAL")
        st.write("Top K: 4")
        st.write("Strategy: MMR")
    
    st.markdown("<h1>Knowledge Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaaaaa !important;'>Ask questions across your documents using grounded AI.</p>", unsafe_allow_html=True)
    
    if st.session_state.vectorstore is None:
        st.info("Please upload PDF documents to get started.")
        st.markdown("""
        **Capabilities:**
        - **Document Grounding:** Answers are solely based on your documents.
        - **Semantic Retrieval:** Find relevant information even without exact keywords.
        - **Source Citations:** Every answer includes exact page references.
        """)
        return
    
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)
            if role == "assistant" and hasattr(msg, "context_chunks") and msg.context_chunks:
                with st.expander("View Retrieved Context"):
                    for i, doc in enumerate(msg.context_chunks):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(doc.page_content)
                        st.markdown("---")

    
    if prompt := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = answer_question(
                        st.session_state.rag_chain, 
                        prompt, 
                        st.session_state.chat_history
                    )
                    
                    answer = result["answer"]
                    sources = result["source_documents"]
                    
                    source_text = ""
                    if "I couldn't find this information" not in answer:
                        source_text = "\n\n**Sources**\n-------\n"
                        unique_sources = set()
                        for doc in sources:
                            source_name = doc.metadata.get("source", "Unknown Document")
                            page = doc.metadata.get("page", 0) + 1 
                            unique_sources.add(f"📄 {source_name} — Page {page}")
                        
                        if unique_sources:
                            for s in sorted(list(unique_sources)):
                                source_text += f"{s}\n"
                    
                    full_response = answer + source_text
                    st.markdown(full_response)
                    
                    with st.expander("View Retrieved Context"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(doc.page_content)
                            st.markdown("---")
                    
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    
                    ai_msg = AIMessage(content=full_response)
                    ai_msg.context_chunks = sources
                    st.session_state.chat_history.append(ai_msg)
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
