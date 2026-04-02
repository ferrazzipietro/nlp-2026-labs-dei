# Lab 1 — Using Hugging Face Transformers

Objective: Learn to use the `transformers` library with encoder-only (BERT) and decoder-only (Llama 1B) models.

## Structure
- Overview of the library
- Import and usage of BERT with `cache_dir`, loading and inference parameters
- Usage of a Gemma chat model with special tokens, tokenization, and chat template
- Exercises:
  1. Generate embeddings for 5 sentences and find the two most similar
  2. Ask Gemma to identify person names in a text using a system prompt: first zero-shot, then three-shot (explain what a "shot" is)

## Repo Layout
- `lab1/notebooks/lab1_transformers.ipynb` — the notebook
- `lab1/data/sentences.txt` — five sentences for similarity
- `lab1/data/target_text.txt` — text for person extraction
- `lab1/data/few_shot_examples.json` — three NER examples for few-shot
- `lab1/models_cache/` — local cache for downloaded models

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r lab1/requirements.txt
```

Optional: configure Hugging Face cache location via `cache_dir` in code or globally via `HF_HOME`.

## Running
Open the notebook and follow cells in order. For the Llama section, consider GPU for speed. The notebook defaults to not generating with Llama (`RUN_LLAMA=False`); flip to `True` when ready and resources allow.

## Notes
- Zero-shot: no examples provided, the model relies only on instructions.
- Few-shot (e.g., three-shot): include a few input-output examples in the prompt to guide the model.
