# 🤖 DocuTube AI

DocuTube AI is a Retrieval-Augmented Generation (RAG) based application that lets users chat with YouTube videos, PDF documents, or both together.
It answers questions strictly using the provided content.

---

##  Features

- Ask questions from YouTube video transcripts
- Ask questions from PDF documents
- Combine YouTube + PDF as one knowledge base
- Session-based chat memory
- Fast semantic search using FAISS
- LLM-powered responses using Groq (LLaMA 3.1)
- Graceful handling of missing transcripts, PDFs, and API failures

---

##  Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq API (LLaMA 3.1)

---

## Project Structure

app.py  
config.py  
loaders.py  
rag_pipeline.py  
requirements.txt  
README.md  

---

## How It Works

1. Load YouTube transcripts and/or PDFs
2. Split content into chunks
3. Generate embeddings
4. Store vectors in FAISS
5. Retrieve relevant chunks
6. Generate answers using Groq LLM

---

