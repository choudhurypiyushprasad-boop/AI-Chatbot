
import chromadb
from sentence_transformers import SentenceTransformer

print("Step 1: Script started")

import chromadb
print("Step 2: chromadb imported")

from sentence_transformers import SentenceTransformer
print("Step 3: sentence-transformers imported")


# Connect to ChromaDB
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_collection("documents")

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def search(query, n_results=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or type 'exit'): ")

        if question.lower() == "exit":
            break

        results = search(question)

        print("\nTop Results:\n")

        for i, doc in enumerate(results["documents"][0], start=1):
            print(f"Result {i}")
            print("-" * 60)
            print(doc)
            print("-" * 60)