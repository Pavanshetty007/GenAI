"""
This module handles PDF ingestion for a Retrieval-Augmented Generation (RAG) pipeline.
It extracts text from uploaded PDF files, splits them into chunks, updates the global state,
and rebuilds retrieval indices.

Functions:
- process_uploaded_files: Reads PDFs, chunks text, and updates dense and sparse retrieval indexes.
"""

import pdfplumber, os
from hashlib import md5
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
from utils.state import state
from utils.logger import get_logger

from .dense_retriever import update_dense_index
from .sparse_retriever import update_bm25_index

logger = get_logger(__name__)

def process_uploaded_files(files):
    """
    Processes a list of uploaded PDF files by extracting text, splitting it into chunks,
    and updating the system's document state and retrieval indexes.

    Steps:
    - Skips already-processed documents using an MD5 hash.
    - Extracts raw text using pdfplumber (one string per file).
    - Splits the text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
    - Updates the global state with:
        - Chunk data per document.
        - Mapping from chunk indices to documents.
    - Rebuilds both dense (TF-IDF) and sparse (BM25) retrieval indexes.

    Args:
        files (List[gr.File]): List of uploaded file objects from Gradio's File component.

    Returns:
        str: A status message indicating how many new documents were processed and the total chunk count.
    """
    if not files:
        logger.warning("process_uploaded_files called with no files.")
        return "No files uploaded."
    new_docs = 0
    try:
        for f in files:
            file_path = f.name
            h = md5(open(file_path,"rb").read()).hexdigest()
            if h in state.processed_documents:
                continue
            # extract & chunk
            try:
                with pdfplumber.open(file_path) as pdf:
                    text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            except Exception as e:
                logger.error(f"Error extracting text from {file_path}: {e}", exc_info=True)
                continue
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, length_function=len
            )
            chunks = splitter.split_text(text)
            state.processed_documents[h] = {"chunks": chunks, "meta": os.path.basename(file_path)}
            start = len(state.all_chunks)
            state.all_chunks.extend(chunks)
            for idx in range(start, len(state.all_chunks)):
                state.chunk_to_doc_map[idx] = h
            new_docs += 1

        # rebuild indices
        update_dense_index()
        update_bm25_index()
    except Exception as e:
        logger.critical(f"Unexpected error in process_uploaded_files: {e}", exc_info=True)
        return f"Error processing files: {e}"
    logger.info(f"Processed {new_docs} new docs; total chunks: {len(state.all_chunks)}")
    return f"✅ Processed {new_docs} new doc(s). Total chunks: {len(state.all_chunks)}"
