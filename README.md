# Advanced RAG & Agentic AI System

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white)

## Project Overview

A production-oriented Retrieval-Augmented Generation system implementing four advanced retrieval strategies — **RAG Fusion**, **HyDE**, **CRAG**, and **Graph RAG** — evaluated against a real-world-style multi-domain web corpus. The system is designed to handle noisy retrieval environments and diverse question types with a modular, pipeline-based architecture.

---

## Key Features

- **Four RAG Pipelines:** RAG Fusion (multi-query retrieval with reciprocal rank fusion), HyDE (hypothetical document embeddings for zero-shot dense retrieval), CRAG (corrective retrieval with relevance grading), and Graph RAG (entity-relation graph traversal for multi-hop reasoning)
- **Global Embedding Index:** All page snippets across the corpus are indexed into a single FAISS vector store; all four pipelines retrieve from this shared index
- **React Frontend:** Interactive UI for selecting pipelines, submitting queries, and comparing outputs side-by-side
- **Modular Backend:** Each RAG strategy is implemented as an isolated, swappable pipeline module

---

## Tech Stack

**Backend**
- Python 3.9+
- LangChain / LangGraph
- FAISS
- Sentence Transformers (`all-MiniLM-L6-v2`)
- Groq / Google Gemini (LLM inference)

**Frontend**
- React (Node.js 18+)
- Streamlit (evaluation dashboard)

---

## Setup

### Backend
```bash
pip install -r requirements.txt
```

Copy `config/config.example.yaml` to `config/config.yaml` and configure:
- `embedding_model` — e.g. `all-MiniLM-L6-v2`
- `generation_model` — LLM for answer generation
- `top_k` — number of chunks retrieved per query

### Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## Running

```bash
# Run evaluation across all pipelines
python run_evaluation.py

# Launch frontend
cd frontend && npm run dev
```

---

## Repository Structure

├── config/
├── dataset/
├── docs/
├── frontend/
├── pipelines/
│   ├── rag_fusion.py
│   ├── hyde.py
│   ├── crag.py
│   └── graph_rag.py
├── run_evaluation.py
└── README.md

---

## Corpus

The system is evaluated against a large multi-domain web search corpus. Each entry contains a `query`, `answer`, `alt_ans`, and `search_results` (up to 5 page snippets). All snippets are embedded into a unified FAISS index at startup.
