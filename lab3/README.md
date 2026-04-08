# Lab 2 — Building a Minimal RAG

Objective: Build a Retrieval-Augmented Generation (RAG) system using components from Lab 1, focusing on understanding internal mechanisms rather than efficiency.

## Structure
- RAG overview and design choices
- Embedder: BERT (encoder-only) for chunk embeddings
- Generator: gemma (decoder-only) for answer generation with context
- Exercises:
  1. Convert documents into a JSON knowledge base (KB) with `text`, `embedding`, and `embedding_dim`
  2. Implement a function to retrieve the top-n most similar chunks for a query
  3. Generate answers using retrieved chunks as context (short answers only, sourced from KB)

## Repo Layout
- `lab2/notebooks/lab2_rag.ipynb` — the notebook
- `lab2/data/kb_docs.json` — 25 short documents about Padua (tourism, work, university, ecclesial)
- `lab2/data/queries.json` — 5 queries with expected short answers, all present in the KB
- `lab2/models_cache/` — local cache for downloaded models

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r lab2/requirements.txt
```

## Running
Open the notebook and follow cells in order.
- Indexing produces `lab2/data/kb_index.json` containing chunk metadata and embeddings.
- Retrieval returns top-n chunks by cosine similarity.
- Generation uses gemma with a concise system prompt and the retrieved context. By default, generation is disabled (`RUN_gemma=False`). Enable only with sufficient resources (preferably GPU).

## RAG Overview
A RAG system retrieves relevant knowledge chunks and feeds them into a generator to produce grounded answers. In this lab:
- Embeddings use pooled hidden states from BERT (`last_hidden_state` mean)
- Retrieval uses cosine similarity
- Prompting instructs the model to answer concisely and only from provided context

## Real-World Implementations and Resources
- Storage: **pgvector** (Postgres extension), **FAISS**, **Milvus**, **Weaviate**, **Pinecone**
- Serving: **vLLM** (open source), closed-source serving options
- Frameworks: **LangGraph**, **gemmaIndex**

## Notes
- Keep answers short and grounded in KB content.
- For evaluation, compare generated answers with the `expected_answer` for each query.
