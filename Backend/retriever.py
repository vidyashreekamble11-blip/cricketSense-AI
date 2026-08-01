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
        print("DEBUG: Retriever already initialized, skipping re-init.")
        return  # Already initialized

    print("=" * 60)
    print("DEBUG: Starting retriever initialization")
    print(f"DEBUG: VECTOR_DB path -> {VECTOR_DB}")
    print(f"DEBUG: VECTOR_DB exists? -> {os.path.isdir(VECTOR_DB)}")
    print("=" * 60)

    print("STEP 1: Loading HuggingFace embeddings...")
    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("DEBUG: HuggingFace embeddings loaded successfully.")

    print("STEP 2: Loading Chroma database...")
    _db = Chroma(
        persist_directory=VECTOR_DB,
        embedding_function=_embeddings
    )
    print("DEBUG: Chroma database object created.")

    # Try to introspect how many documents/embeddings are actually in the DB
    try:
        collection_data = _db.get()
        num_docs = len(collection_data.get("ids", []))
        print(f"DEBUG: Chroma collection reports {num_docs} stored documents.")
        if num_docs == 0:
            print("DEBUG: WARNING - The vector DB collection appears to be EMPTY!")
    except Exception as e:
        print(f"DEBUG: Could not introspect Chroma collection: {e}")

    print("STEP 3: Creating retriever...")
    _retriever = _db.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )
    print("DEBUG: Retriever created with search_type='similarity', k=4")
    print("STEP 4: Initialization complete!")


def _mmr_fallback_search(question):
    """Fallback MMR search with a larger fetch_k, used if similarity search fails."""
    print("DEBUG: Attempting MMR fallback search (fetch_k=50, lambda_mult=0.7)...")
    mmr_retriever = _db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 4,
            "fetch_k": 50,
            "lambda_mult": 0.7
        }
    )
    docs = mmr_retriever.invoke(question)
    print(f"DEBUG: MMR fallback search returned {len(docs)} documents.")
    return docs


def search_laws(question):
    """Search for relevant laws. Initializes on first call."""
    _initialize()

    print("\n" + "=" * 60)
    print(f"DEBUG: Searching for question -> '{question}'")
    print("=" * 60)

    # Log similarity scores for visibility, regardless of what the retriever returns
    try:
        scored_docs = _db.similarity_search_with_score(question, k=4)
        print(f"DEBUG: similarity_search_with_score returned {len(scored_docs)} results.")
        for i, (doc, score) in enumerate(scored_docs):
            print(f"DEBUG: [Score Check] Doc {i + 1} | score={score:.4f}")
            print(f"DEBUG: [Score Check] Doc {i + 1} preview: {doc.page_content[:200]!r}")
    except Exception as e:
        print(f"DEBUG: similarity_search_with_score failed: {e}")

    print("DEBUG: Invoking primary retriever (similarity search)...")
    docs = _retriever.invoke(question)
    print(f"DEBUG: Primary retriever returned {len(docs)} documents.")

    if not docs:
        print("DEBUG: Primary retriever returned 0 docs. Trying plain similarity_search fallback...")
        try:
            docs = _db.similarity_search(question, k=4)
            print(f"DEBUG: similarity_search fallback returned {len(docs)} documents.")
        except Exception as e:
            print(f"DEBUG: similarity_search fallback failed: {e}")
            docs = []

    if not docs:
        print("DEBUG: similarity_search fallback also returned 0 docs. Trying MMR fallback...")
        try:
            docs = _mmr_fallback_search(question)
        except Exception as e:
            print(f"DEBUG: MMR fallback failed: {e}")
            docs = []

    if not docs:
        print("DEBUG: WARNING - All search strategies returned 0 documents!")
        print("DEBUG: This likely means the vector DB is empty or the embeddings")
        print("DEBUG: used to query do not match the embeddings used to build the DB.")
    else:
        print(f"DEBUG: Final result -> {len(docs)} documents will be returned.")
        for i, doc in enumerate(docs):
            print(f"DEBUG: Final Doc {i + 1} preview: {doc.page_content[:200]!r}")

    return docs
