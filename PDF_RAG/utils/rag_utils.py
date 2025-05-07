"""
This module implements the main logic for the Hybrid RAG (Retrieval-Augmented Generation) system.

Features:
- Chat history management with timestamps.
- Knowledge Graph (KG) entity lookup.
- Substring-based fallback for technical queries (e.g., formulas).
- Hybrid retrieval using sparse and dense methods.
- Text generation using Ollama.
- Document and index clearing functionality.

Dependencies:
- ollama: for language model inference.
- datetime: for timestamp formatting.
- utils.state: global in-memory state.
- config: model and generation parameters.
- utils.logger: centralized logging.
"""

import ollama
import os
from datetime import datetime
from utils.state       import state
from utils.logger      import get_logger
from .hybrid_retriever import retrieve_hybrid
from .kg_utils         import query_kg
from config            import (
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
    OLLAMA_NUM_CTX
)

logger = get_logger(__name__)

def _now():
    """Return current timestamp as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _format_message(role: str, content: str) -> dict:
    """
    Format a chat message with a timestamp in Markdown.

    Args:
        role (str): 'user' or 'assistant'.
        content (str): The message content.

    Returns:
        dict: Formatted message with a timestamp.
    """
    ts = _now()
    markdown = f"{content}\n\n*_{ts}_*"
    return {"role": role, "content": markdown}

def generate_response(query: str, chunks: list[tuple[str,str]]) -> str:
    """
    Use the Ollama language model to generate a response based on retrieved document chunks.

    Args:
        query (str): User's question.
        chunks (list of tuple): Context chunks and their sources.

    Returns:
        str: Assistant's generated response or error message.
    """
    prompt = """<s>[INST] <<SYS>>
    You are an articulate AI assistant that provides:
    1. Contextually precise answers using ONLY the provided references
    2. Well-structured responses 
    If context is insufficient, respond: "The provided references don't contain relevant information."

    References:
    {context}
    <</SYS>>

    Question: {query} [/INST]""".format(
            context="\n\n".join(f"---\nContent: {c}\nSource: {s}" for c, s in chunks),
            query=query
        )

    try:
        resp = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={
                "temperature": OLLAMA_TEMPERATURE,
                "top_p":       OLLAMA_TOP_P,
                "num_ctx":     OLLAMA_NUM_CTX
            }
        )
        logger.info("Ollama response generated for query: %s", query)
        return resp["response"].strip()
    except Exception as e:
        logger.error("Error generating response for query '%s': %s", query, e, exc_info=True)
        return f"Error generating response: {e}"

def substring_fallback(query: str) -> list[tuple[str,str]] | None:
    """
    If key technical terms (e.g., formulas) are detected in the query,
    return matching chunks directly as fallback.

    Args:
        query (str): User's question.

    Returns:
        Optional[List[Tuple[str, str]]]: Matched context chunks or None if no match.
    """
    key_terms = ["PE(", "positional encoding", "sin(", "cos("]
    try:
        if any(term.lower() in query.lower() for term in key_terms):
            matches = []
            for chunk in state.all_chunks:
                if any(term in chunk for term in key_terms):
                    matches.append((chunk, ""))
                    if len(matches) >= state.max_chat_history:
                        break
            logger.info("Substring fallback triggered for query: %s, %d matches", query, len(matches))
            return matches
    except Exception as e:
        logger.error("Error in substring_fallback for '%s': %s", query, e, exc_info=True)
    return None

def rag_chat(message: str, history: list[dict]) -> list[dict]:
    """
    Main entry point for RAG-based chat interaction.

    Process flow:
    1. Append user message to history.
    2. Attempt Knowledge Graph lookup.
    3. If no KG result, try formula substring fallback.
    4. Otherwise, use hybrid retrieval.
    5. Generate and append assistant response.
    6. Trim history to the most recent N exchanges.

    Args:
        message (str): User's chat input.
        history (list of dict): Chat history, each message as a dict.

    Returns:
        list of dict: Updated chat history.
    """
    try:
        if not message.strip():
            return history
    
        # 1) Append user message
        history.append(_format_message("user", message))
        logger.debug("User message appended: %s", message)

        # 2) KG lookup
        kg_ans = query_kg(message)
        if kg_ans:
            history.append(_format_message("assistant", f"KG Lookup: {kg_ans}"))
            logger.info("KG lookup success for query: %s", message)
        else:
            # 3) Substring fallback for formulas
            fb = substring_fallback(message)
            if fb is not None:
                chunks = fb
                logger.debug("Using substring fallback chunks.")
            else:
                # 4) Hybrid sparse + dense retrieval
                chunks = retrieve_hybrid(message)
                logger.debug("Using hybrid retrieval, %d chunks retrieved.", len(chunks))

            # 5) Generate the assistant answer
            answer = generate_response(message, chunks)
            history.append(_format_message("assistant", answer))
            logger.info("Assistant response appended.")

        # 6) Trim to last N exchanges (2 messages per exchange)
        max_msgs = state.max_chat_history * 2
        if len(history) > max_msgs:
            history = history[-max_msgs:]
            logger.debug("Trimmed history to last %d messages.", max_msgs)

    except Exception as e:
        logger.error("Error in rag_chat for message '%s': %s", message, e, exc_info=True)
        # Ensure the user still sees an error message
        history.append(_format_message("assistant", f"An error occurred: {e}"))

    return history

def clear_documents() -> str:
    """
    Reset the entire RAG system:
    - Clears loaded PDFs, all chunks, retrieval indices, and the knowledge graph.

    Returns:
        str: Confirmation message.
    """
    try:
        state.processed_documents.clear()
        state.all_chunks.clear()
        state.chunk_to_doc_map.clear()
        state.vectorizer       = None
        state.chunk_embeddings = None
        state.bm25_index       = None
        if hasattr(state, "kg"):
            state.kg = None
        logger.info("Cleared all documents and indices from state.")
        return "All documents cleared."
    except Exception as e:
        logger.error("Error clearing documents: %s", e, exc_info=True)
        return f"Error during clear: {e}"
