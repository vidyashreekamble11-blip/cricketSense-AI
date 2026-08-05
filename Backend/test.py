from retriever import search_laws

question = input("Ask: ")

docs = search_laws(question)

print("\nRESULTS\n")

for i, doc in enumerate(docs):

    print("=" * 60)

    print(f"Result {i+1}")

    print("Score:", doc["score"])

    print(doc["text"][:700])