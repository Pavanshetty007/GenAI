"""
Hybrid retriever module.

Performs a combined sparse (BM25) and dense (TF-IDF) retrieval to rank document chunks.
"""

from .sparse_retriever import retrieve_sparse
from .dense_retriever import retrieve_dense
from config import TOP_K, SPARSE_WEIGHT, DENSE_WEIGHT
from utils.logger import get_logger

logger = get_logger(__name__)

def retrieve_hybrid(query):
    """
    Performs hybrid retrieval of relevant text chunks based on a combination of sparse and dense methods.

    This function retrieves candidate chunks using both sparse (e.g., BM25) and dense (e.g., TF-IDF or embeddings) methods.
    Each chunk is assigned a weighted score based on its rank in each retrieval type. The final list of top-ranked
    chunks is determined by combining these scores, giving a hybrid approach to ranking.

    The weights and number of results are controlled by global config variables:
    - `TOP_K`: number of final results to return
    - `SPARSE_WEIGHT`: contribution of the sparse retriever to the final score
    - `DENSE_WEIGHT`: contribution of the dense retriever to the final score

    Args:
        query (str): The user's query to search relevant chunks for.

    Returns:
        List[Tuple[str, str]]: A list of top-ranked text chunks as tuples with an empty source string.
                               Returns an empty list on error.
    """
    try:
        # Retrieve extended candidates
        sparse_hits = retrieve_sparse(query, TOP_K * 2)
        dense_hits  = retrieve_dense(query, TOP_K * 2)
        logger.info("Hybrid retrieval: got %d sparse hits and %d dense hits for query '%s'.",
                    len(sparse_hits), len(dense_hits), query)

        # Combine scores
        scores = {}
        for rank, (chunk, _) in enumerate(sparse_hits):
            scores[chunk] = SPARSE_WEIGHT * (TOP_K * 2 - rank)
        for rank, (chunk, _) in enumerate(dense_hits):
            scores[chunk] = scores.get(chunk, 0) + DENSE_WEIGHT * (TOP_K * 2 - rank)

        # Select top_k
        best = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        results = [(chunk, "") for chunk, _ in best]
        logger.info("Hybrid retrieval: returning %d top results for query '%s'.", len(results), query)
        return results

    except Exception as e:
        logger.error("Error in hybrid retrieval for query '%s': %s", query, e, exc_info=True)
        return []
