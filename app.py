import os
from dotenv import load_dotenv

import chromadb
from google import genai
from sentence_transformers import SentenceTransformer


# ==============================
# LOAD ENVIRONMENT VARIABLES
# ==============================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")


# ==============================
# GEMINI CLIENT
# ==============================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


# ==============================
# EMBEDDING MODEL
# ==============================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==============================
# CONNECT TO CHROMADB
# ==============================

client = chromadb.PersistentClient(path="vector_store")

collection = client.get_collection("documents")


# ==============================
# RETRIEVE RELEVANT DOCUMENTS
# ==============================

def retrieve(query, n_results=3):

    query_embedding = embedding_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]

    return documents


# ==============================
# ASK GEMINI
# ==============================

def ask_gemini(question):

    documents = retrieve(question)

    if not documents:
        return "I could not find relevant information in the knowledge base.", []

    context = "\n\n--- DOCUMENT CHUNK ---\n\n".join(documents)

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the information provided
in the context below.

If the answer is not present in the context, say:

"I could not find that information in the provided documents."

Do not make up information.

Context:
{context}

Question:
{question}
"""

    print("\nDEBUG: Using Gemini model = gemini-3.5-flash")

    response = gemini_client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    return response.text, documents


# ==============================
# MAIN CHATBOT
# ==============================

if __name__ == "__main__":

    print("=" * 50)
    print("📚 RAG Chatbot Started")
    print("=" * 50)

    while True:

        question = input("\nYou: ")

        if question.lower() == "exit":
            print("\nChatbot stopped.")
            break

        try:

            answer, sources = ask_gemini(question)

            print("\nBot:", answer)

            print("\n📖 Sources used:")

            for i, source in enumerate(sources, 1):

                preview = source.replace("\n", " ")[:200]

                print(f"\n[{i}] {preview}...")

        except Exception as e:

            print("\n❌ Error:", e)