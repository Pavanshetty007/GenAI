import json
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def store_data_in_chroma(processed_data):
    """
    Stores processed IFC data in ChromaDB.

    Args:
        processed_data (list): A list of dictionaries containing processed IFC data.

    Raises:
        Exception: If data cannot be stored in ChromaDB.
    """
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_db = Chroma(
            collection_name="ifc_data",
            persist_directory="chroma_db",
            embedding_function=embedding_model
        )

        for entry in processed_data:
            project_name = entry.get("project_name", "Unknown Project")
            file_name = entry.get("file_name", "Unknown File")
            materials = ", ".join(entry.get("materials", []))
            spatial_info = json.dumps(entry.get("spatial_info", {}))
            element_counts = json.dumps(entry.get("element_counts", {}))
            quantities = json.dumps(entry.get("quantities", {}))

            metadata = {
                "project_name": project_name,
                "file_name": file_name,
                "materials": materials,
                "spatial_info": spatial_info,
                "element_counts": element_counts,
                "quantities": quantities,
            }

            doc_text = json.dumps(entry)
            vector_db.add_texts(texts=[doc_text], metadatas=[metadata])
            logging.info(f"✅ Stored: {file_name} -> Project: {project_name}")

        logging.info("IFC data successfully stored in ChromaDB!")
    except Exception as e:
        logging.error(f"Error storing data: {e}")
        raise