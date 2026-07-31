from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_database():

    print("Loading PDF...")

    loader = PyPDFLoader("../knowledge/Laws-of-Cricket-2017-Code-3rd-Edition-2022_1.pdf")

    documents = loader.load()

    print("Splitting PDF into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.split_documents(documents)

    print(f"Total Chunks: {len(docs)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Creating Vector Database...")

    Chroma.from_documents(
        docs,
        embeddings,
        persist_directory="../vector_db"
    )

    print("Vector Database Created Successfully!")


if __name__ == "__main__":
    create_vector_database()