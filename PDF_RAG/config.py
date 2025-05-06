"""
Configuration module for environment variables and application constants.

This module sets default values for all major settings used in the RAG system,
including chunking parameters, retrieval weights, chat settings, and Ollama model parameters.
These can be overridden using environment variables at runtime.
"""

import os

# === PDF Chunking Settings ===
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))          # Max tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))    # Overlap between chunks for context preservation

# === Retrieval Settings ===
TOP_K = int(os.getenv("TOP_K", 3))                      # Number of top results to retrieve
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", 0.3))  # Weight for BM25-based sparse retriever
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", 0.7))    # Weight for TF-IDF dense retriever

# === Chat Settings ===
MAX_EXCHANGES = int(os.getenv("MAX_EXCHANGES", 5))      # Number of recent Q&A pairs retained in chat history

# === Ollama LLM Configuration ===
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")              # LLM model name (e.g., llama2, gemma3)
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", 0.7))  # Controls randomness of output
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", 0.9))              # Top-p sampling cutoff
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", 4096))           # Maximum context window size
