from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

print(">>> Step 1: Starting retriever")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(">>> Step 2: Embeddings loaded")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB = os.path.join(BASE_DIR, "vector_db")

print(">>> Step 3: Vector DB path =", VECTOR_DB)
print(">>> Exists?", os.path.exists(VECTOR_DB))

db = Chroma(
    persist_directory=VECTOR_DB,
    embedding_function=embeddings
)

print(">>> Step 4: Chroma loaded")

retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,
        "lambda_mult": 0.7
    }
)

print(">>> Step 5: Retriever ready")

def search_laws(question):
    return retriever.invoke(question)