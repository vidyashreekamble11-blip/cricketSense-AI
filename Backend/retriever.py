print("STEP 1")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("STEP 2")

db = Chroma(
    persist_directory=VECTOR_DB,
    embedding_function=embeddings
)

print("STEP 3")