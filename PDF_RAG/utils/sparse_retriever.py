"""
Sparse retriever module using Whoosh and BM25.

This script builds and queries a BM25-based index over all document chunks
stored in the application state. The index is rebuilt from scratch each time
new documents are processed.

Main Functions:
- update_bm25_index: Builds and stores a Whoosh index for sparse retrieval.
- retrieve_sparse: Retrieves top-k relevant chunks from the index for a query.

Dependencies:
- Whoosh: full-text indexing and BM25 ranking
- utils.state: shared in-memory state holding chunk data
- utils.logger: centralized logging
"""
import os
import shutil
from whoosh import index, fields, writing, qparser
from whoosh.analysis import StemmingAnalyzer
from utils.state import state
from utils.logger import get_logger
from config import TOP_K

logger = get_logger(__name__)

def update_bm25_index():
    """
    Create or rebuild a BM25 index over all text chunks using Whoosh.

    Steps:
    1. Removes the old index directory (if any).
    2. Defines a schema with stemming for better search matching.
    3. Adds all document chunks from the global state.
    4. Stores the index back to global state.

    Returns:
        None
    """
    index_dir = "bm25_index"
    try:
        # 1) Remove old index to pick up schema changes
        if os.path.exists(index_dir):
            shutil.rmtree(index_dir)
            logger.debug("Deleted existing BM25 index directory.")
        os.makedirs(index_dir, exist_ok=True)
        logger.debug("Created BM25 index directory: %s", index_dir)

        # 2) Define schema with content stored
        schema = fields.Schema(
            id=fields.NUMERIC(stored=True),
            content=fields.TEXT(stored=True, analyzer=StemmingAnalyzer())
        )

        # 3) Create new index
        idx = index.create_in(index_dir, schema)
        logger.info("Created new Whoosh BM25 index in %s", index_dir)

        # 4) Add documents
        writer = idx.writer()
        writer.mergetype = writing.CLEAR
        for i, chunk in enumerate(state.all_chunks):
            writer.add_document(id=i, content=chunk)
        writer.commit()
        logger.info("Indexed %d chunks into BM25 index.", len(state.all_chunks))

        # 5) Save to state
        state.bm25_index = idx
        logger.info("BM25 index updated in application state.")

    except Exception as e:
        logger.error("Failed to update BM25 index: %s", e, exc_info=True)


def retrieve_sparse(query, top_k=TOP_K):
    """
    Perform sparse retrieval using BM25 over the indexed document chunks.

    Args:
        query (str): User input query.
        top_k (int, optional): Number of top results to retrieve. Defaults to TOP_K.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing matched text chunks
                               and their source indices. Returns an empty list
                               on error or if index is unavailable.
    """
    if state.bm25_index is None:
        logger.warning("retrieve_sparse called but bm25_index is None.")
        return []

    try:
        idx = state.bm25_index
        qp = qparser.QueryParser("content", schema=idx.schema)
        q  = qp.parse(query)
        with idx.searcher() as searcher:
            results = searcher.search(q, limit=top_k)
            hits = [(hit["content"], f"Source idx={hit['id']}") for hit in results]
            logger.info("retrieve_sparse for query '%s' returned %d hits.", query, len(hits))
            return hits

    except Exception as e:
        logger.error("BM25 search error for query '%s': %s", query, e, exc_info=True)
        return []
