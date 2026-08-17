import os
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "arxiv" / "data" / "arxiv_cs.json"

VECTOR_STORE = str(
    BASE_DIR / "arxiv_vector_store"
)

COLLECTION_NAME = "arxiv_papers"

TOP_K = 5


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found.\n"
        "Make sure your .env file contains:\n\n"
        "GEMINI_API_KEY=your_api_key_here"
    )


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

genai.configure(
    api_key=GEMINI_API_KEY
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("🧠 Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# LOAD PAPERS
# ============================================================

def load_papers():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"arXiv dataset not found:\n{DATA_FILE}"
        )

    with open(
        DATA_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        papers = json.load(file)

    print(
        f"📚 Loaded {len(papers)} papers"
    )

    return papers


# ============================================================
# CREATE / LOAD CHROMADB
# ============================================================

def create_database(papers):

    print("💾 Creating ChromaDB...")

    client = chromadb.PersistentClient(
        path=VECTOR_STORE
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    existing_count = collection.count()

    if existing_count > 0:

        print(
            f"ℹ️ Database already contains "
            f"{existing_count} papers"
        )

        return collection

    print("🧠 Creating embeddings...")

    documents = []
    ids = []
    metadatas = []

    for index, paper in enumerate(papers):

        title = paper.get(
            "title",
            ""
        )

        abstract = paper.get(
            "abstract",
            ""
        )

        document = (
            f"Title: {title}\n\n"
            f"Abstract: {abstract}"
        )

        documents.append(document)

        ids.append(
            f"paper_{index}"
        )

        authors = paper.get(
            "authors",
            []
        )

        categories = paper.get(
            "categories",
            []
        )

        metadatas.append(
            {
                "title": title,

                "authors": ", ".join(
                    authors
                ),

                "categories": ", ".join(
                    categories
                ),

                "published": paper.get(
                    "published",
                    ""
                ),

                "arxiv_id": paper.get(
                    "id",
                    ""
                ),

                "arxiv_url": paper.get(
                    "url",
                    ""
                )
            }
        )

    embeddings = embedding_model.encode(
        documents,
        show_progress_bar=True
    )

    print("💾 Storing papers in ChromaDB...")

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"✅ Stored {len(documents)} papers"
    )

    return collection


# ============================================================
# SEARCH PAPERS
# ============================================================

def search_papers(
    collection,
    query,
    top_k=TOP_K
):

    query_embedding = embedding_model.encode(
        [query]
    )[0]

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return results


# ============================================================
# BUILD RESEARCH CONTEXT
# ============================================================

def build_context(results):

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    if not documents:

        return "No relevant research papers were found."

    context_parts = []

    for index, document in enumerate(
        documents
    ):

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        distance = (
            distances[index]
            if index < len(distances)
            else None
        )

        title = metadata.get(
            "title",
            "Unknown"
        )

        authors = metadata.get(
            "authors",
            "Unknown"
        )

        categories = metadata.get(
            "categories",
            "Unknown"
        )

        published = metadata.get(
            "published",
            "Unknown"
        )

        arxiv_id = metadata.get(
            "arxiv_id",
            "Unknown"
        )

        section = f"""
PAPER {index + 1}

Title:
{title}

Authors:
{authors}

Categories:
{categories}

Published:
{published}

arXiv ID:
{arxiv_id}

Retrieval Distance:
{distance}

Research Content:
{document}
"""

        context_parts.append(
            section
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE RESEARCH ANSWER
# ============================================================

def generate_research_answer(
    question,
    collection
):

    results = search_papers(
        collection,
        question,
        top_k=TOP_K
    )

    context = build_context(
        results
    )

    prompt = f"""
You are an AI research assistant
specialized in Computer Science.

Answer the user's question using the
research papers retrieved from the arXiv
knowledge base.

IMPORTANT RULES:

1. Base your answer primarily on the
   provided research context.

2. Do not invent facts, results,
   experiments, authors, or citations.

3. If the retrieved papers do not contain
   enough information to answer something,
   clearly say that the available papers
   do not provide enough evidence.

4. Explain technical concepts clearly.

5. When useful, compare approaches
   between papers.

6. Mention relevant paper titles when
   making claims.

7. Give a concise but useful answer.

8. Distinguish evidence from your own
   interpretation.

9. Do not claim that a paper proves
   something unless the provided research
   context supports that claim.

10. Use Markdown headings and bullet
    points where appropriate.

RETRIEVED RESEARCH
==================

{context}

USER QUESTION
=============

{question}

Now provide the research-based answer.
"""

    # --------------------------------------------------------
    # Gemini model
    # --------------------------------------------------------

    model = genai.GenerativeModel(
        "gemini-3.6-flash"
    )

    response = model.generate_content(
        prompt
    )

    return response.text, results


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]

    if not metadatas:

        return

    print()
    print("=" * 60)
    print("📚 SOURCES USED")
    print("=" * 60)

    for index, metadata in enumerate(
        metadatas
    ):

        print()
        print(
            f"{index + 1}. "
            f"{metadata.get('title', 'Unknown')}"
        )

        print(
            f"   Authors: "
            f"{metadata.get('authors', 'Unknown')}"
        )

        print(
            f"   Categories: "
            f"{metadata.get('categories', 'Unknown')}"
        )

        print(
            f"   Published: "
            f"{metadata.get('published', 'Unknown')}"
        )

        print(
            f"   arXiv ID: "
            f"{metadata.get('arxiv_id', 'Unknown')}"
        )

        if metadata.get("arxiv_url"):

            print(
                f"   URL: "
                f"{metadata.get('arxiv_url')}"
            )

        if index < len(distances):

            print(
                f"   Distance: "
                f"{distances[index]}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("🔬 ARXIV RESEARCH Q&A ASSISTANT")
    print("=" * 60)

    print()

    print(
        "📚 Loading research knowledge base..."
    )

    papers = load_papers()

    collection = create_database(
        papers
    )

    print()

    print(
        f"📊 Papers available: "
        f"{collection.count()}"
    )

    print()

    print(
        "Ask a question about the research."
    )

    print(
        "Type 'exit' to quit."
    )

    # ========================================================
    # CONTINUOUS QUESTION LOOP
    # ========================================================

    while True:

        question = input(
            "\nResearch question: "
        ).strip()

        if not question:

            print(
                "⚠️ Please enter a question."
            )

            continue

        if question.lower() in {
            "exit",
            "quit",
            "q"
        }:

            print(
                "\n👋 Exiting research assistant."
            )

            break

        print()

        print(
            "🧠 Retrieving research and "
            "generating answer..."
        )

        try:

            answer, results = (
                generate_research_answer(
                    question,
                    collection
                )
            )

            print()
            print("=" * 60)
            print("💡 RESEARCH ANSWER")
            print("=" * 60)

            print()
            print(answer)

            display_sources(
                results
            )

        except Exception as e:

            print()
            print(
                "❌ Error generating answer:"
            )

            print(e)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()