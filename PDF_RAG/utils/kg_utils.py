"""
This module constructs and queries a lightweight knowledge graph (KG) from processed document chunks.
It uses spaCy for Named Entity Recognition (NER) and RDFLib for creating and querying the graph.

Functions:
- build_kg: Extracts named entities from document chunks and builds the KG.
- query_kg: Looks up entities from a user question in the KG.

If required NLP or RDF libraries are unavailable, fallback no-op implementations are used.
"""

from utils.state import state
from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import spacy
    from rdflib import Graph, URIRef, Literal

    # Load SpaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        logger.info("Loaded spaCy model 'en_core_web_sm'.")
    except OSError as e:
        logger.error("spaCy model 'en_core_web_sm' not found: %s", e, exc_info=True)
        raise

    kg = Graph()

    def build_kg():
        """
        Constructs a knowledge graph (KG) from named entities found in processed document chunks.

        This function uses spaCy's NER to extract entities from each document chunk and populates
        an RDFLib graph with triples of the form:
            (entity_text, has_label, entity_type)

        The graph is stored in `state.kg` for later querying.

        Dependencies:
            - Requires spaCy with the 'en_core_web_sm' model.
            - Requires RDFLib.

        Returns:
            None
        """
        try:
            kg.remove((None, None, None))
            for h, doc in state.processed_documents.items():
                for chunk in doc["chunks"]:
                    doc_nlp = nlp(chunk)
                    for ent in doc_nlp.ents:
                        kg.add((URIRef(ent.text), URIRef("has_label"), Literal(ent.label_)))
            state.kg = kg
            logger.info("Knowledge graph built with %d triples.", len(kg))
        except Exception as e:
            logger.error("Error building knowledge graph: %s", e, exc_info=True)

    def query_kg(question):
        """
        Queries the knowledge graph for any entities mentioned in the user's question.

        For each token in the question, this function checks if it exists as a subject in the graph.
        If found, it retrieves and returns all associated object values (typically entity types).

        Args:
            question (str): The user's natural language question.

        Returns:
            str or None: A comma-separated string of object values for matched entities,
                         or None if no match is found or the KG is not built.
        """
        if state.kg is None:
            logger.warning("query_kg called but state.kg is None.")
            return None
        try:
            for token in question.split():
                subj = URIRef(token)
                if (subj, None, None) in kg:
                    objs = [str(o) for o in kg.objects(subj, None)]
                    result = ", ".join(objs)
                    logger.info("query_kg found %d objects for token '%s'.", len(objs), token)
                    return result
        except Exception as e:
            logger.error("Error querying knowledge graph for '%s': %s", question, e, exc_info=True)
        return None

except (ImportError, OSError):
    # Fallback if spaCy or RDFLib is unavailable
    def build_kg():
        """Fallback no-op if KG dependencies are missing."""
        logger.warning("build_kg no-op: KG dependencies unavailable.")

    def query_kg(question):
        """Fallback no-op query, always returns None."""
        logger.warning("query_kg no-op: KG dependencies unavailable.")
        return None
