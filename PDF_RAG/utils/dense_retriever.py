"""
Dense retriever module using TF-IDF and cosine similarity.

Builds and queries a dense vector index over all document chunks stored in the application state.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils.state import state
from utils.logger import get_logger

logger = get_logger(__name__)

def update_dense_index():
    """
    Builds a TF-IDF vector index for all text chunks in the application state.

    This function initializes a TfidfVectorizer with English stop words and fits it to the list of all
    text chunks stored in the application's state. The resulting vectorizer and the transformed chunk
    embeddings are stored back into the state for future retrieval operations.

    Returns:
        None
    """
    try:
        if not state.all_chunks:
            logger.warning("update_dense_index called with no chunks in state.")
            return
        state.vectorizer = TfidfVectorizer(stop_words="english")
        state.chunk_embeddings = state.vectorizer.fit_transform(state.all_chunks)
        logger.info("Dense index built with %d chunks.", len(state.all_chunks))
    except Exception as e:
        logger.error("Failed to build dense index: %s", e, exc_info=True)
        # Clean up partial state
        state.vectorizer = None
        state.chunk_embeddings = None

def retrieve_dense(query, top_k):
    """
    Retrieves the top_k most relevant text chunks to the given query using cosine similarity.

    This function transforms the input query into a TF-IDF vector using the pre-fitted vectorizer from the
    application state. It then computes the cosine similarity between the query vector and all stored
    chunk embeddings, returning the top_k chunks with the highest similarity scores.

    Args:
        query (str): The input query string to search for relevant chunks.
        top_k (int): The number of top similar chunks to retrieve.

    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing a text chunk and a string indicating its source index.
                               Returns an empty list on error or if the index is unavailable.
    """
    if state.vectorizer is None or state.chunk_embeddings is None:
        logger.warning("retrieve_dense called but dense index is not available.")
        return []

    try:
        qv = state.vectorizer.transform([query])
        sims = cosine_similarity(qv, state.chunk_embeddings).flatten()
        idxs = sims.argsort()[-top_k:][::-1]
        results = [(state.all_chunks[i], f"Source idx={i}") for i in idxs]
        logger.info("retrieve_dense for query '%s' returned %d hits.", query, len(results))
        return results
    except Exception as e:
        logger.error("Dense retrieval error for query '%s': %s", query, e, exc_info=True)
        return []
