# Mini AI Knowledge Assistant

A professional, local Retrieval-Augmented Generation (RAG) application to chat with your PDFs.

## Overview
This project is a high-contrast, black-and-white styled Streamlit application that allows users to upload PDF documents, index them, and ask questions using a grounded AI approach. It strictly relies on the provided documents to answer and explicitly refuses to fabricate information or citations.

## Features
- **Document Grounding**: Answers are formulated purely from uploaded context.
- **Semantic Retrieval**: Uses Sentence Transformers and ChromaDB to fetch relevant context using MMR.
- **Source Citations**: Clearly references source documents and pages for each answer.
- **Conversation History**: Supports multi-turn questions and follow-ups.

## Architecture
- **Text Extraction**: PyPDF
- **Chunking**: RecursiveCharacterTextSplitter (chunk_size=800, overlap=120)
- **Embeddings**: `all-MiniLM-L6-v2` via HuggingFace
- **Vector Database**: ChromaDB
- **LLM**: Google Gemini (`gemini-1.5-flash`) via LangChain

## Setup & Installation

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and provide your Google API Key:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` to include your `GOOGLE_API_KEY`.

## Run Locally
To start the application, run:
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

## Limitations & Future Improvements
- The vector store is persisted locally and clears out when uploading a fresh batch.
- Future improvements could include caching embeddings for faster re-indexing.

## AI Disclosure
This project was implemented with the assistance of an AI coding assistant.
