# Enterprise Knowledge Assistant (RAG + FAISS + Groq LLM)

An end-to-end Retrieval-Augmented Generation (RAG) system that allows users to query PDFs and internal documents using semantic search, vector embeddings, and a grounded LLM (Llama 3.3 via Groq).

This project was built from scratch (without heavy RAG frameworks) to deeply understand the internals of:
- Chunking
- Embeddings
- Vector Search (FAISS)
- Context Grounding
- LLM Integration

---

## Project Overview

Traditional LLMs hallucinate because they rely only on pre-trained knowledge.

This system solves that by:
1. Retrieving relevant document chunks using FAISS semantic search
2. Injecting retrieved context into the LLM prompt
3. Generating grounded, factual responses based only on source documents

This mimics how enterprise AI assistants are built in production environments.

---

## Key Features

- PDF Document Ingestion (pdfminer)
- Semantic Chunking (Paragraph-aware)
- Embeddings using Sentence Transformers (MiniLM)
- Vector Search with FAISS (Cosine Similarity)
- Fast LLM Inference using Groq (Llama-3.3-70B)
- Anti-Hallucination Grounded Prompting
- Source Transparency (Shows retrieved chunks + scores)
- Optional Streamlit UI for interactive querying


---

##  Tech Stack

- Python 3.x
- FAISS (Vector Indexing)
- Sentence Transformers (Embeddings)
- pdfminer.six (PDF Parsing)
- Groq API (Llama-3.3-70B)
- Streamlit (UI)
- NumPy / Scikit-learn

---

##  Core Concepts Implemented (Deep Learning Focus)

### 1. Semantic Chunking
Instead of naive character slicing, documents are chunked by paragraphs to preserve semantic meaning and improve retrieval accuracy.

### 2. Embeddings
Text chunks are converted into 384-dimensional vectors using `all-MiniLM-L6-v2` for semantic similarity search.

### 3. Vector Search (FAISS)
FAISS IndexFlatIP with normalized embeddings enables efficient cosine similarity-based retrieval.

### 4. Grounded RAG Prompting
The LLM is forced to answer ONLY using retrieved context to reduce hallucinations.

---

##  Example Workflow

**Query:**  
> "What is Retrieval-Augmented Generation?"

**Pipeline Execution:**
- Query embedded into vector
- Top-K relevant chunks retrieved from FAISS
- Context injected into LLM prompt
- Groq LLM generates grounded answer
