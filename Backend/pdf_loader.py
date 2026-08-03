import os
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve paths relative to the repository root so this works both locally
# (where the repo root is the parent of Backend/) and in deployment, where
# Railway runs the app from /app (the repo root).
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
PDF_PATH = os.path.join(
    KNOWLEDGE_DIR, "Laws-of-Cricket-2017-Code-3rd-Edition-2022_1.pdf"
)
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")


def vector_db_exists():
    """Check whether a populated vector DB already exists on disk."""
    if not os.path.isdir(VECTOR_DB_PATH):
        return False

    # A freshly-created but empty directory should not count as "existing".
    return len(os.listdir(VECTOR_DB_PATH)) > 0


def create_vector_database(force=False):
    """Build the Chroma vector database from the PDFs in knowledge/.

    This is idempotent: if the vector DB already exists at VECTOR_DB_PATH it
    will not be rebuilt unless force=True is passed.
    """

    if not force and vector_db_exists():
        logger.info(
            "Vector DB already exists at %s, skipping creation.", VECTOR_DB_PATH
        )
        return

    logger.info("Vector DB not found (or force=True). Building vector database...")

    if not os.path.isfile(PDF_PATH):
        logger.error("PDF not found at expected path: %s", PDF_PATH)
        raise FileNotFoundError(f"PDF not found at expected path: {PDF_PATH}")

    try:
        logger.info("Loading PDF from %s ...", PDF_PATH)
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        logger.info("Loaded %d page(s) from PDF.", len(documents))

        logger.info("Splitting PDF into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)
        logger.info("Total chunks: %d", len(docs))

        if not docs:
            raise ValueError("No document chunks were produced from the PDF.")

        logger.info("Loading HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        os.makedirs(VECTOR_DB_PATH, exist_ok=True)

        logger.info("Creating vector database at %s ...", VECTOR_DB_PATH)
        Chroma.from_documents(
            docs,
            embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        logger.info("Vector database created successfully at %s", VECTOR_DB_PATH)

    except Exception:
        logger.exception("Failed to create vector database.")
        raise


if __name__ == "__main__":
    create_vector_database()
