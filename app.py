import ollama
import chromadb
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader
import streamlit as st
from typing import List
import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex

from llama_index.llms.ollama import Ollama


chroma_client = chromadb.EphemeralClient()
chroma_collection = chroma_client.get_or_create_collection("mydocs")

#llamaindex connectors
#Chroma vector store , that will contain query and doc embedding
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# context contains all docs close to query 
storage_context = StorageContext.from_defaults(vector_store=vector_store)

#Load and explore documents
documents = SimpleDirectoryReader("docs", recursive=True).load_data()

#Embedding Model from HuggingFace
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
)
    
llm = Ollama(model="llama3.2", request_timeout=120.0)

def query_rag(my_query):
    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=3,
        streaming=True,
    )
    response = query_engine.query(my_query)
    response_text = ""
    for chunk in response.response_gen:
        response_text += chunk
    return response_text



st.set_page_config(page_title="RAG Application", layout="wide")

st.title("Welcome to your Enhanced RAG Application!")

# Sidebar for configuration
with st.sidebar:
    st.header("App Settings")
    similarity_top_k = st.slider("Number of similar documents to retrieve", 1, 10, 3)
    st.markdown("Upload documents in the main section to update the knowledge base.")

# Main layout
query_container = st.container()
response_container = st.container()

with query_container:
    st.subheader("Query the Knowledge Base")
    user_input = st.text_input("Enter your question:")

if st.button("Submit"):
    if user_input:
        st.write("Processing your query...")
        result = query_rag(user_input)  # Call the query function with the selected model and similarity
        with response_container:
            st.subheader("Response")
            st.text(result)
    else:
        st.warning("Please enter some text before submitting!")

