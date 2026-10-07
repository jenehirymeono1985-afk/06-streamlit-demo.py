import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DATA_DIR = Path("data/handbook")


def get_api_key():
    """Validates that the Gemini API key is loaded from the .env file."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY. Please ensure it is added to your .env file.")
        st.stop()
    return api_key


def validate_data_directory():
    """Checks if the data directory exists and contains valid files."""
    if not DATA_DIR.exists() or not DATA_DIR.is_dir():
        st.error(f"Data directory not found. The app expected a folder named: {DATA_DIR}")
        st.stop()

    files = [f for f in os.listdir(DATA_DIR) if not f.startswith('.')]
    if not files:
        st.error(f"The data directory '{DATA_DIR}' is empty. Please add your assignment documents.")
        st.stop()


@st.cache_resource
def get_query_engine(api_key):
    """Loads documents, configures the AI models, and builds the search index."""
    Settings.llm = GoogleGenAI(model="models/gemini-3.8-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    documents = SimpleDirectoryReader(input_dir=str(DATA_DIR)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


st.title("Bare Bones RAG Chatbot")

api_key = get_api_key()
validate_data_directory()

try:
    query_engine = get_query_engine(api_key)
except Exception as e:
    st.error(f"Failed to build the search engine. Please check your data files or model connection. Error: {e}")
    st.stop()

prompt = st.chat_input("Ask me anything...")
if prompt:
    st.write(f"User: {prompt}")
    try:
        response = query_engine.query(prompt)
        st.write(response.response)
    except Exception as e:
        st.error("There was a network or connection error when processing your question. Please try again later!")