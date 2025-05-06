import streamlit as st
import os
import tempfile
import shutil
import zipfile
from dotenv import load_dotenv
from preprocess import extract_ifc_data
from store import store_data_in_chroma
from chatbot import query_llm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import logging
import json
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
quant_config = BitsAndBytesConfig(load_in_4bit=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=quant_config,
    token=HF_TOKEN
)

# Streamlit app
st.title("IFC RAG Chatbot")

# Section 1: Upload ZIP and Preprocess
st.header("1. Upload IFC ZIP Folder and Preprocess")
uploaded_zip = st.file_uploader("Upload a ZIP folder containing IFC files", type=["zip"])

if st.button("Preprocess and Store"):
    if not uploaded_zip:
        st.error("Please upload a ZIP folder.")
    else:
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, "uploaded.zip")

        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.getbuffer())

        extracted_folder = os.path.join(temp_dir, "extracted_files")
        os.makedirs(extracted_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extracted_folder)

        processed_data = []
        processed_files = []

        # Traverse the extracted folder recursively
        for root, _, files in os.walk(extracted_folder):
            for filename in files:
                if filename.lower().endswith(".ifc"):  # Case-insensitive check
                    file_path = os.path.join(root, filename)
                    try:
                        data = extract_ifc_data(file_path)
                        processed_data.append(data)
                        processed_files.append(filename)
                        logging.info(f"Processed file: {filename}")
                    except Exception as e:
                        logging.error(f"Error processing {filename}: {e}")
                        st.error(f"Error processing {filename}: {e}")

        if processed_data:
            store_data_in_chroma(processed_data)
            st.success(f"Preprocessing and storing complete! Processed {len(processed_files)} files.")
            st.write("Processed files:", processed_files)

            # Add a download button for processed data
            json_data = json.dumps(processed_data, indent=4)
            st.download_button(
                label="Download Processed Data (JSON)",
                data=json_data,
                file_name="processed_ifc_data.json",
                mime="application/json"
            )

            # Visualize element counts
            st.write("### Element Counts")
            element_counts = processed_data[0].get("element_counts", {})  # Use the first file's data
            if element_counts:
                fig = px.bar(
                    x=list(element_counts.keys()),
                    y=list(element_counts.values()),
                    labels={"x": "Element Type", "y": "Count"},
                    title="Element Counts in IFC File"
                )
                st.plotly_chart(fig)
        else:
            st.error("No valid IFC files found in the uploaded ZIP folder.")

        shutil.rmtree(temp_dir)

# Section 2: LLM Query
st.header("2. Ask a Query")
query = st.text_input("Enter your query about the IFC data:")

if st.button("Submit Query"):
    if not query:
        st.error("Please enter a query.")
    else:
        try:
            answer = query_llm(query, tokenizer, model, device)
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"Error during query: {e}")