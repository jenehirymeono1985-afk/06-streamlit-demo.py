import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core import Settings
import streamlit as st

load_dotenv()

# Step 2: Define constant for your data directory
DATA_DIR = "data/handbook"


# Step 3: Fail Fast Validations
def get_api_key():
    """Validates that the Gemini API key is loaded from the .env file."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Missing GEMINI_API_KEY. Please ensure it is set in your .env file.")
        st.stop()
    return api_key


def validate_data_directory():
    """Checks that the data directory exists, is a folder, and contains non-hidden files."""
    data_path = Path(DATA_DIR)

    if not data_path.exists() or not data_path.is_dir():
        st.error(f"Error: The expected data folder '{DATA_DIR}' does not exist.")
        st.stop()

    # Check for files, ignoring hidden ones like Mac's .DS_Store
    files = [f for f in data_path.iterdir() if not f.name.startswith('.')]
    if not files:
        st.error(f"Error: The folder '{DATA_DIR}' is empty. Please add your documents.")
        st.stop()


# Step 4: Caching Optimizations
@st.cache_resource
def get_query_engine(api_key):
    """Loads documents, sets models, and builds the query engine index."""
    # Settings moved inside the cached function, passing the validated key
    Settings.llm = GoogleGenAI(model="gemini-3.8-flash", api_key=api_key)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # Using the DATA_DIR constant here
    documents = SimpleDirectoryReader(DATA_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()


st.title("Bare Bones Rag Chatbot")

# Run validations before doing anything else
validated_key = get_api_key()
validate_data_directory()

# Step 5: Handle Runtime Errors - Engine Build (Fail fast)
try:
    # Pass the validated key into the engine
    query_engine = get_query_engine(validated_key)
except Exception as e:
    st.error(
        f"Failed to build the AI engine. The documents might be corrupted or the model failed to download. Error: {e}")
    st.stop()

prompt = st.chat_input("Ask me anything...")
if prompt:
    st.write(f"User: {prompt}")

    # Step 5: Handle Runtime Errors - Asking a question (Fallback gracefully)
    try:
        response = query_engine.query(prompt)
        bot_response = response.response
        with st.chat_message("assistant"):
            st.write(f"Bot response: {bot_response}")
    except Exception as e:
        # We show an error but DO NOT use st.stop() so the user can try asking again
        st.error(f"There was a network or connection error when processing your question. Please try again! Error: {e}")