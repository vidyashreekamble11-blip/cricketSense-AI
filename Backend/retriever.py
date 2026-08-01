import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DB = os.path.join(BASE_DIR, "vector_db")

# Lazy-loading: initialize only on first use
_embeddings = None
_db = None
_retriever = None


def _initialize():
    """Initialize embeddings, db, and retriever on first use."""
    global _embeddings, _db, _retriever
    
    if _retriever is not None:
        return  # Already initialized
    
    print("STEP 1: Loading HuggingFace embeddings...")
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    print("STEP 2: Loading Chroma database...")
    _db = Chroma(
        persist_directory=VECTOR_DB,
        embedding_function=_embeddings
    )
    
    print("STEP 3: Creating retriever...")
    _retriever = _db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )
    print("STEP 4: Initialization complete!")


def search_laws(question):
    """Search for relevant laws. Initializes on first call."""
    _initialize()
    docs = _retriever.invoke(question)
    return docs

