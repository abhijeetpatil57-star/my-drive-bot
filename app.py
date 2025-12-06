
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import io
import streamlit as st
from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pypdf import PdfReader

# --- ROBUST IMPORTS (Handles Version Conflicts) ---
try:
    # Try the new v0.3 structure first
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    # Fallback to the old structure
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredPowerPointLoader
# -------------------------------------------------- 

# 2. Load Environment Variables
load_dotenv()

# Force Google Credentials path
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")

st.set_page_config(page_title="DriveBot (Gemini + Local)", page_icon="📂")
st.title("📂 Chat (Gemini Free + Local Embeddings)")

# 3. Validation
if not os.getenv("GOOGLE_API_KEY"):
    st.error("❌ GOOGLE_API_KEY is missing. Check your .env file.")
    st.stop()

# --- CUSTOM LOADER FUNCTION ---
def load_local_files():
    docs_accumulator = []
    
    # 1. Load PDFs
    st.write("📂 Reading local PDFs...")
    pdf_loader = DirectoryLoader('./my_docs', glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs_accumulator.extend(pdf_loader.load())

    # 2. Load Text/Docs
    st.write("📂 Reading local Text files...")
    txt_loader = DirectoryLoader('./my_docs', glob="**/*.txt", loader_cls=TextLoader)
    docs_accumulator.extend(txt_loader.load())
    
    # (Optional) Load PowerPoints - Requires: pip install unstructured networkx
    # ppt_loader = DirectoryLoader('./my_docs', glob="**/*.pptx", loader_cls=UnstructuredPowerPointLoader)
    # docs_accumulator.extend(ppt_loader.load())

    return docs_accumulator

@st.cache_resource
def build_knowledge_base():
    folder_id = os.getenv("GOOGLE_DRIVE_FOLDER_ID")
    #credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
    
    # Load files locally
    docs = load_local_files()
    if not docs: return None
    
    st.success(f"✅ Loaded {len(docs)} files successfully.")

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # --- LOCAL EMBEDDINGS (HuggingFace) ---
    # This runs on your CPU. It is free and avoids the "Quota 0" error.
    st.info("🧠 Creating Embeddings locally (This might take 30s the first time)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(splits, embeddings)
    
    return vectorstore

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am ready to chat about your files."}]

vectorstore = build_knowledge_base()

# Chat Interface
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not vectorstore: st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # --- GEMINI CHAT MODEL ---
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    retriever = vectorstore.as_retriever()
    
    system_prompt = (
        "You are a helpful assistant. Answer based strictly on the context provided."
        "\n\n"
        "{context}"
    )
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )
    chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt_template))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": prompt})["answer"]
            st.write(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
