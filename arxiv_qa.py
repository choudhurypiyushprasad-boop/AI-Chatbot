import os
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from transformers import pipeline
from google import genai


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
# ENVIRONMENT
# ============================================================

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

print("🤖 Initializing Gemini client...")

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# SENTIMENT MODEL
# ============================================================

print("🧠 Loading sentiment analysis model...")

sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("🧠 Loading embedding model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text):

    result = sentiment_model(
        text,
        truncation=True
    )[0]

    label = result["label"].lower()
    confidence = float(result["score"])

    if label == "positive":
        sentiment = "positive"

    elif label == "negative":
        sentiment = "negative"

    else:
        sentiment = "neutral"

    return sentiment, confidence


# ============================================================
# SENTIMENT RESPONSE INSTRUCTIONS
# ============================================================

def get_sentiment_instruction(sentiment):

    if sentiment == "positive":

        return """
The user has a positive sentiment.

Respond in a friendly, warm, and encouraging way.
You may acknowledge their positive feedback briefly.
Do not become excessively enthusiastic or distract
from the user's actual question.
"""

    elif sentiment == "negative":

        return """
The user has a negative or frustrated sentiment.

Respond calmly and empathetically.
Acknowledge the user's concern when appropriate.
Do not sound defensive, dismissive, or overly cheerful.
If the user reports an error or incorrect answer,
focus on correcting the problem.
"""

    return """
The user's sentiment is neutral.

Respond clearly, professionally, and directly.
Do not add unnecessary emotional language.
"""


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

    client_db = chromadb.PersistentClient(
        path=VECTOR_STORE
    )

    collection = client_db.get_or_create_collection(
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

        title = paper.get("title", "")
        abstract = paper.get("abstract", "")

        document = (
            f"Title: {title}\n\n"
            f"Abstract: {abstract}"
        )

        documents.append(document)

        ids.append(
            f"paper_{index}"
        )

        metadatas.append(
            {
                "title": title,
                "authors": ", ".join(
                    paper.get("authors", [])
                ),
                "categories": ", ".join(
                    paper.get("categories", [])
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
# BUILD CONTEXT
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

    for index, document in enumerate(documents):

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

        context_parts.append(
            f"""
PAPER {index + 1}

Title:
{metadata.get("title", "Unknown")}

Authors:
{metadata.get("authors", "Unknown")}

Categories:
{metadata.get("categories", "Unknown")}

Published:
{metadata.get("published", "Unknown")}

arXiv ID:
{metadata.get("arxiv_id", "Unknown")}

Retrieval Distance:
{distance}

Research Content:
{document}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GENERATE RESEARCH ANSWER
# ============================================================

def generate_research_answer(
    question,
    collection,
    sentiment
):

    results = search_papers(
        collection,
        question,
        top_k=TOP_K
    )

    context = build_context(results)

    sentiment_instruction = get_sentiment_instruction(
        sentiment
    )

    prompt = f"""
You are an AI research assistant specialized
in Computer Science.

{sentiment_instruction}

Answer the user's question using the research
papers retrieved from the arXiv knowledge base.

IMPORTANT RULES:

1. Base factual research claims primarily on
   the provided research context.

2. Do not invent facts, results, experiments,
   authors, or citations.

3. If the papers do not contain enough
   information, clearly say so.

4. Explain technical concepts clearly.

5. When useful, compare approaches between papers.

6. Mention relevant paper titles when making
   research claims.

7. Distinguish evidence from interpretation.

8. Do not claim that a paper proves something
   unless the provided context supports it.

9. Use Markdown headings and bullet points
   where appropriate.

10. Keep the response proportional to the user's
    question. Do not produce a long research
    report for a simple conversational message.

RETRIEVED RESEARCH
==================

{context}

USER QUESTION
=============

{question}

Detected sentiment:
{sentiment}

Now provide the best response.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
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

    for index, metadata in enumerate(metadatas):

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

        try:

            # ------------------------------------------------
            # SENTIMENT
            # ------------------------------------------------

            print()
            print(
                "🧠 Analyzing sentiment..."
            )

            sentiment, confidence = analyze_sentiment(
                question
            )

            emoji = {
                "positive": "😊",
                "negative": "😟",
                "neutral": "😐"
            }.get(
                sentiment,
                "😐"
            )

            print()
            print("=" * 60)
            print("😊 SENTIMENT ANALYSIS")
            print("-" * 60)

            print(
                f"Sentiment: {emoji} {sentiment}"
            )

            print(
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            )

            # ------------------------------------------------
            # RESEARCH / CONVERSATIONAL DETECTION
            # ------------------------------------------------

            research_keywords = [
                "paper",
                "papers",
                "research",
                "arxiv",
                "model",
                "models",
                "algorithm",
                "algorithms",
                "transformer",
                "transformers",
                "machine learning",
                "deep learning",
                "neural network",
                "neural networks",
                "vision",
                "llm",
                "llms",
                "retrieval",
                "rag",
                "dataset",
                "datasets",
                "accuracy",
                "architecture",
                "inference",
                "training"
            ]

            is_research_question = any(
                keyword in question.lower()
                for keyword in research_keywords
            )

            # ------------------------------------------------
            # CONVERSATIONAL MESSAGE
            # ------------------------------------------------

            if not is_research_question:

                sentiment_instruction = (
                    get_sentiment_instruction(
                        sentiment
                    )
                )

                prompt = f"""
You are a friendly AI assistant.

{sentiment_instruction}

Respond naturally to the user's message.

Do not retrieve or discuss research papers
unless the user asks about research.

USER MESSAGE:
{question}
"""

                print()
                print(
                    "💬 Conversational message detected."
                )

                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )

                print()
                print("=" * 60)
                print("💡 ASSISTANT RESPONSE")
                print("=" * 60)
                print()
                print(response.text)

                continue

            # ------------------------------------------------
            # RESEARCH QUESTION
            # ------------------------------------------------

            print()
            print(
                "🧠 Retrieving research and "
                "generating answer..."
            )

            answer, results = (
                generate_research_answer(
                    question,
                    collection,
                    sentiment
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
                "❌ Error:"
            )
            print(e)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()