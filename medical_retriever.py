import chromadb
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIGURATION
# ==============================

VECTOR_DB_PATH = "medical_vector_store"
COLLECTION_NAME = "medquad"
TOP_K = 5

# ==============================
# LOAD EMBEDDING MODEL
# ==============================

print("🧠 Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# CONNECT TO MEDQUAD DATABASE
# ==============================

client = chromadb.PersistentClient(
    path=VECTOR_DB_PATH
)

collection = client.get_collection(
    COLLECTION_NAME
)

# ==============================
# RETRIEVAL FUNCTION
# ==============================

def retrieve_medical_info(question, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved.append({
            "document": document,
            "focus": metadata.get("focus", ""),
            "question": metadata.get("question", ""),
            "source": metadata.get("source", ""),
            "distance": distance
        })

    return retrieved


# ==============================
# TEST RETRIEVER
# ==============================

if __name__ == "__main__":

    print("=" * 60)
    print("🏥 MEDQUAD MEDICAL RETRIEVER")
    print("=" * 60)

    question = input(
        "\nEnter a medical question: "
    )

    results = retrieve_medical_info(question)

    print("\n🔎 Retrieved medical information:")
    print("=" * 60)

    for i, result in enumerate(results, 1):

        print(f"\nRESULT {i}")
        print("-" * 60)

        print(
            f"Medical Topic: {result['focus']}"
        )

        print(
            f"Dataset Question: {result['question']}"
        )

        print(
            f"Distance: {result['distance']:.4f}"
        )

        print("\nAnswer:")
        print(result["document"])