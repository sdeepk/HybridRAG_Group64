# Hybrid RAG System – Assignment 2

This repository implements a **Hybrid Retrieval-Augmented Generation (RAG)** system using **Dense Retrieval (ChromaDB)**, **Sparse Retrieval (BM25)**, **Reciprocal Rank Fusion (RRF)**, and **Flan-T5** for answer generation.  
It also includes a **fully automated evaluation pipeline** with mandatory and custom metrics.

---

## Installation Steps

1. Clone or download the repository.
2. Navigate to the project root directory.
3. (Recommended) Create and activate a Python virtual environment.
4. Install all required dependencies using pip.

---

## Dependencies

Install the required libraries:

```bash
pip install streamlit pandas numpy matplotlib tqdm scikit-learn \
torch transformers sentence-transformers chromadb rank-bm25 nltk

```

---

## Run Instructions

### 1. Run the Hybrid RAG System (Interactive)

The standalone interface is implemented using **Streamlit**.

```bash
streamlit run streamlit_app.py
```

This opens the application in the browser at:

```
http://localhost:8501
```

The interface allows:
- Entering natural language queries
- Viewing retrieved documents 
- Viewing generated answers from Flan-T5

---

### 2. Run the Automated Evaluation Pipeline

The complete evaluation pipeline can be executed using a **single command**:

```bash
python src/metrics/metrics_pipeline.py
```

This pipeline:
- Loads the evaluation questions
- Runs hybrid retrieval and answer generation
- Computes all metrics (MRR, Precision@5, Faithfulness)
- Saves results and visualizations

Evaluation outputs are written to:

```
data/eval_run/
```

---

## Fixed Wikipedia URLs (200 URLs)

The fixed set of **200 Wikipedia URLs** used for corpus creation and evaluation is stored in JSON format at:

```
data/fixed_urls.json
```

This file is used consistently across:
- Data preprocessing
- Index creation
- Question generation
- Evaluation

Using a fixed URL list ensures **reproducibility and fair evaluation**.

---

## Summary

- Hybrid RAG system with Dense + Sparse retrieval and RRF
- Automated evaluation with MRR, Precision@5, and Faithfulness
- Streamlit-based interactive interface
- Fully reproducible using fixed Wikipedia URLs

---
