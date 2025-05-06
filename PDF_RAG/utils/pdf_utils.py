# utils/pdf_utils.py

import os
import shutil
import pdfplumber
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
    Processes uploaded PDF files:
    - Copies each PDF into `static/` for serving
    - Safely extracts text page-by-page, skipping corrupted/unreadable PDFs
    - Splits text into chunks with page tracking
    - Rebuilds dense and sparse indices
    - Returns detailed per-file status messages for the UI
    """
    if not files:
        logger.warning("No files provided to process_uploaded_files.")
        return "No files uploaded."

    os.makedirs("static", exist_ok=True)
    processed_count = 0
    messages = []

    for f in files:
        orig_path   = f.name
        filename    = os.path.basename(orig_path)
        static_path = os.path.join("static", filename)

        # 1) Copy PDF to static for serving
        try:
            shutil.copy(orig_path, static_path)
            logger.info("Copied PDF %s to static folder.", filename)
        except Exception as e:
            msg = f"❌ Failed to copy {filename}: {e}"
            logger.error(msg, exc_info=True)
            messages.append(msg)
            continue

        # 2) Compute document hash
        try:
            with open(orig_path, "rb") as fh:
                file_hash = md5(fh.read()).hexdigest()
        except Exception as e:
            msg = f"❌ Failed to hash {filename}: {e}"
            logger.error(msg, exc_info=True)
            messages.append(msg)
            continue

        if file_hash in state.processed_documents:
            msg = f"ℹ️ Skipping already-processed {filename}"
            logger.info(msg)
            messages.append(msg)
            continue

        # 3) Extract text, skip any read errors
        try:
            text = ""
            with pdfplumber.open(orig_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\f"
            logger.info("Extracted text from %s", filename)
        except Exception as e:
            msg = f"⚠️ Skipping unreadable PDF {filename}: {e}"
            logger.warning(msg, exc_info=True)
            messages.append(msg)
            continue

        # 4) Chunk text with page tracking
        try:
            pages        = text.split("\f")
            all_chunks   = []
            page_indices = []
            splitter     = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                length_function=len
            )
            for pi, pg in enumerate(pages):
                chunks = splitter.split_text(pg)
                all_chunks.extend(chunks)
                page_indices.extend([pi] * len(chunks))
            logger.info("Split %s into %d chunks", filename, len(all_chunks))
        except Exception as e:
            msg = f"❌ Error chunking {filename}: {e}"
            logger.error(msg, exc_info=True)
            messages.append(msg)
            continue

        # 5) Update global state
        try:
            state.processed_documents[file_hash] = {
                "chunks": all_chunks,
                "pages":  page_indices,
                "meta":   static_path
            }
            start_idx = len(state.all_chunks)
            state.all_chunks.extend(all_chunks)
            for idx in range(start_idx, len(state.all_chunks)):
                state.chunk_to_doc_map[idx] = file_hash
            processed_count += 1
            msg = f"✅ Processed {filename} ({len(all_chunks)} chunks)"
            logger.info(msg)
            messages.append(msg)
        except Exception as e:
            msg = f"❌ Error updating state for {filename}: {e}"
            logger.error(msg, exc_info=True)
            messages.append(msg)
            continue

    # 6) Rebuild retrieval indices
    try:
        update_dense_index()
        update_bm25_index()
        summary = f"🏁 Done: {processed_count} new docs, {len(state.all_chunks)} total chunks"
        logger.info(summary)
        messages.append(summary)
    except Exception as e:
        msg = f"❌ Failed to rebuild indices: {e}"
        logger.critical(msg, exc_info=True)
        messages.append(msg)

    return "\n".join(messages)
