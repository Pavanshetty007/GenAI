import json
import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def query_llm(query, tokenizer, model, device):
    """
    Queries the LLM with the given query and context from ChromaDB.

    Args:
        query (str): The user's query.
        tokenizer: The tokenizer for the LLM.
        model: The LLM model.
        device (str): The device to run the model on (e.g., "cuda" or "cpu").

    Returns:
        str: The LLM's response.

    Raises:
        Exception: If the query cannot be processed.
    """
    try:
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        collection = chroma_client.get_or_create_collection(name="ifc_data")

        def flatten(nested):
            """
            Flattens a nested list.

            Args:
                nested: A nested list.

            Returns:
                list: A flattened list.
            """
            if isinstance(nested, list):
                flat = []
                for item in nested:
                    flat.extend(flatten(item))
                return flat
            else:
                return [nested]

        all_data = collection.get()
        all_metas = flatten(all_data.get("metadatas", []))
        where_filter = {}
        for meta in all_metas:
            file_name = meta.get("file_name", "").lower()
            if file_name and file_name in query.lower():
                where_filter = {"file_name": meta.get("file_name")}
                break

        if where_filter:
            search_results = collection.query(query_texts=[query], n_results=3, where=where_filter)
        else:
            search_results = collection.query(query_texts=[query], n_results=3)

        retrieved_docs = flatten(search_results.get("documents", []))
        retrieved_metadata = flatten(search_results.get("metadatas", []))

        if not retrieved_docs or not retrieved_metadata:
            return "❌ No relevant IFC data found."

        retrieved_contexts = []
        for i, doc in enumerate(retrieved_docs):
            try:
                doc_data = json.loads(doc)
                if isinstance(doc_data, list):
                    doc_data = next((d for d in doc_data if isinstance(d, dict)), {})

                meta = retrieved_metadata[i] if i < len(retrieved_metadata) else {}
                project_name = meta.get("project_name", doc_data.get("project_name", "Unknown Project"))
                file_name = meta.get("file_name", doc_data.get("file_name", "Unknown File"))
                materials = meta.get("materials", ", ".join(doc_data.get("materials", [])))
                element_counts = meta.get("element_counts", json.dumps(doc_data.get("element_counts", {}), indent=2))
                spatial_info = meta.get("spatial_info", json.dumps(doc_data.get("spatial_info", {}), indent=2))
                quantities = meta.get("quantities", json.dumps(doc_data.get("quantities", {}), indent=2))

                doc_summary = (
                    f"📂 **Project:** {project_name} ({file_name})\n"
                    f"🔹 **Materials Used:** {materials}\n"
                    f"🏗 **Element Counts:** {element_counts}\n"
                    f"📍 **Spatial Info:** {spatial_info}\n"
                    f"📏 **Quantities:** {quantities}\n"
                )
                retrieved_contexts.append(doc_summary)
            except Exception as e:
                logging.error(f"Error processing document: {e}")
                continue

        if not retrieved_contexts:
            return "❌ No relevant IFC data found."

        retrieved_context = "\n\n".join(retrieved_contexts)
        max_context_chars = 1000
        if len(retrieved_context) > max_context_chars:
            retrieved_context = retrieved_context[:max_context_chars] + "\n... [truncated]"

        prompt = (
            f"Context:\n{retrieved_context}\n\n"
            f"User asked: '{query}'. Provide ONLY the relevant information based on the stored IFC data.\n\n"
            f"AI Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            temperature=0.7,
            do_sample=True
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during query: {e}")
        raise