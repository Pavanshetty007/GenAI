# 📚 PDF Document Query Application 

*A Retrieval‑Augmented Generation (RAG) system over multiple PDF documents, featuring BM25 + TF‑IDF hybrid retrieval, Knowledge Graph lookup and timestamped chat history.*

---

## Table of Contents

1. [Features](#features)  
2. [Architecture & Approach](#architecture--approach)  
3. [Assumptions](#assumptions)  
4. [Prerequisites](#prerequisites)
5. [Project Structure](#project-structure)  
6. [Installation](#installation)  
7. [Running Ollama Locally (CPU)](#running-ollama-locally-cpu)  
8. [Usage](#usage)     
9. [Troubleshooting](#troubleshooting)  

---

## Features

- **Multi‑PDF Upload & Indexing** via PDFPlumber & LangChain chunking  
- **Hybrid Retrieval** (BM25 sparse + TF‑IDF dense) with tunable weights  
- **Knowledge Graph** using spaCy NER + RDFLib for factual lookups  
- **Substring Fallback** for exact‑match formulas (e.g. positional encoding)    
- **Chat History** limited to last 5 exchanges, with Markdown timestamps  
- **Single Scrollbar UI** in Gradio Chatbot  

---

## Architecture & Approach

1. **Ingestion**  
   - PDFs are uploaded and copied into `static/`.  
   - Text is extracted page‑by‑page, delimited by `\f`, then chunked with overlap.   

2. **Indexing**  
   - **Dense index**: TF‑IDF vectorizer over chunks → cosine similarity retrieval.  
   - **Sparse index**: Whoosh BM25 index → keyword‑based retrieval.  

3. **Hybrid Retrieval**  
   - Retrieve top 2×TOP_K candidates from both methods.  
   - Score each chunk by weighted sum of sparse & dense ranks.  
   - Select top TOP_K chunks for context.  

4. **Knowledge Graph (KG)**  
   - spaCy NER → RDFLib triples `(entity, has_label, label)`.  
   - If a query token matches an entity, return KG result instead of RAG.  

5. **Fallback**  
   - Queries with “sin(”, “PE(”, etc. trigger substring fallback.    

6. **UI**  
   - Gradio chat interface with single scrollbar & Markdown timestamps.  

---

## Assumptions

- **Local Ollama** provides a CPU‑based LLM (`gemma3`)—no GPU needed.  
- **PDFs** fit in memory; default chunk size = 500, overlap = 100.  
- **Environment**: Windows/Linux, Python 3.9+  

---

## Prerequisites

- Python 3.9+  
- [Ollama](https://ollama.com/) installed locally  
- No GPU required  
- miniconda installed
- Terminal/PowerShell access  

---

## project-structure
```
pdf_rag_app/
├── run.py                    
├── app.py                    
├── config.py                 
├── requirements.txt          
├── static/                   
│   ├── user.png              
│   └── bot.png               
└── utils/                   
    ├── logger.py             
    ├── state.py              
    ├── pdf_utils.py          
    ├── sparse_retriever.py    
    ├── dense_retriever.py    
    ├── hybrid_retriever.py    
    ├── kg_utils.py            
    ├── keywords.py           
    └── rag_utils.py          
```

## Ollama Installation
- Install Ollama per https://ollama.com/docs/install
- Use any model which has best accuracy (I have used opensource model gemma3 since i have hardware constraints)
```bash
ollama pull gemma3

ollama list
```
## Installation

```bash
git clone https://github.com/yourusername/adova-rag-chat.git
cd adova-rag-chat

# Create & activate a conda environment
conda create -n rag-chat python=3.10 -y
conda activate rag-chat

# Install dependencies
python.exe -m pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

```
## Usage
```bash
python run.py
```
- Open your browser at http://127.0.0.1:7860
- Upload PDFs in the left panel
- Click Process PDFs (repeat after adding new docs)
- Ask questions in the chat box & click Send
- The last 5 exchanges are retained by default for context.
- Use the "Clear Chat" button to reset the conversation thread.
## Troubleshooting
- Whoosh index errors:

```bash
rm -rf bm25_index
```
  then re‑process PDFs.

- Ollama not running:

```bash
ollama list
# If empty, pull a model:
ollama pull gemma3
```

- spaCy model missing:

```bash
python -m spacy download en_core_web_sm
```