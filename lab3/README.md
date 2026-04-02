# Lab 3 — Named Entity Recognition (NER)

Objective: Showcase how to perform NER with the Transformers library using two approaches and evaluate results.

## Contents
- BERT-based NER (token classification via `dslim/bert-base-NER`)
- Gemma prompting for NER (chat model returning JSON PERSON/ORG/LOC)
- Evaluation: micro and per-label precision/recall/F1 based on exact text match
- Exercise to extend and compare methods

## Repo Layout
- `lab3/notebooks/lab3_ner.ipynb` — the notebook
- `lab3/data/ner_examples.json` — 10 sentences with gold PERSON/ORG/LOC entities
- `lab3/data/few_shot_ner_examples.json` — 3 examples for few-shot prompting
- `lab3/models_cache/` — cache directory for model artifacts

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r lab3/requirements.txt
```

## Running
Open the notebook and run cells in order.
- BERT pipeline runs on CPU/GPU automatically.
- Gemma generation is disabled by default (`RUN_GEMMA=False`). Enable only if you have sufficient resources (preferably GPU).

## Notes
- Labels are mapped to PERSON/ORG/LOC for consistency.
- Evaluation performs entity-level matching by exact surface text (case-insensitive). Span-level scoring can be added if desired.

## Exercise
- Add types (DATE/EVENT), expand the dataset, and compare zero-shot vs three-shot prompting.
- Improve prompts and report errors with examples.
