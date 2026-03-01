"""
ingest.py – E-Commerce RAG Chatbot
────────────────────────────────────────────────────────────────────────
Loads PDF documents from the documents/ folder, splits them into semantic
chunks, embeds them with OpenAI Embeddings, and stores the vector index
in a local FAISS index (ecommerce_faiss_index/).

Run once (or whenever documents are updated):
    python ingest.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
load_dotenv()

DATA_FOLDER   = "documents"
VECTOR_DB_PATH = "ecommerce_faiss_index"

# ─────────────────────────────────────────────
# Step 1 – Load PDF documents
# ─────────────────────────────────────────────
print("\n[1/4] Loading PDFs from documents/ ...")

pdf_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".pdf")]

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found in the 'documents/' folder.\n"
        "Run 'python create_documents.py' first to generate sample documents."
    )

documents = []
for file in pdf_files:
    path = os.path.join(DATA_FOLDER, file)
    loader = PyPDFLoader(path)
    pages = loader.load()
    documents.extend(pages)
    print(f"   Loaded: {file}  ({len(pages)} pages)")

print(f"\n   Total pages loaded: {len(documents)}")

# ─────────────────────────────────────────────
# Step 2 – Split into semantic chunks
# ─────────────────────────────────────────────
print("\n[2/4] Splitting into text chunks ...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
    separators=["\n\n", "\n", ".", "!", "?", " "],
    length_function=len,
)

chunks = text_splitter.split_documents(documents)
print(f"   Total chunks created: {len(chunks)}")

# ─────────────────────────────────────────────
# Step 3 – Generate embeddings
# ─────────────────────────────────────────────
print("\n[3/4] Generating embeddings with OpenAI text-embedding-3-small ...")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.getenv("OPENAI_API_BASE"),
)

# ─────────────────────────────────────────────
# Step 4 – Store in FAISS vector store
# ─────────────────────────────────────────────
print("\n[4/4] Indexing into FAISS and saving locally ...")

vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(VECTOR_DB_PATH)

print(f"\n   Vector store saved to: {VECTOR_DB_PATH}/")
print("\n✅  E-Commerce knowledge base successfully indexed!")
print("   You can now run: python chatbot.py\n")
