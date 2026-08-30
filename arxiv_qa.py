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

VECTOR_STORE = str(BASE_DIR / "arxiv_vector_store")

COLLECTION_NAME = "arxiv_papers"

TOP_K = 5


# ============================================================
# SUPPORTED LANGUAGES - TASK 6
# ============================================================

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "ko": "Korean",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
}


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "\nGEMINI_API_KEY was not found.\n\n"
        f"Expected .env file at:\n{ENV_FILE}\n\n"
        "Create the .env file with:\n\n"
        "GEMINI_API_KEY=your_api_key_here\n"
        "GEMINI_MODEL=gemini-3.6-flash\n"
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

print("🤖 Initializing Gemini client...")

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

print("🧠 Loading sentiment analysis model...")

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest"
)


# ============================================================
# LANGUAGE DETECTION
# ============================================================

print("🌍 Loading language detection model...")

language_detector = pipeline(
    "text-classification",
    model="papluca/xlm-roberta-base-language-detection"
)


# ============================================================
# EMBEDDING MODEL
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
            f"\n❌ arXiv dataset not found:\n{DATA_FILE}\n"
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
                "authors": ", ".join(authors),
                "categories": ", ".join(categories),
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
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):

    if not text.strip():
        return "en"

    try:

        result = language_detector(
            text[:512]
        )[0]

        detected = result["label"].lower()

        confidence = float(
            result["score"]
        )

        if detected in {
            "zh",
            "zh-cn",
            "zh-tw"
        }:
            detected = "zh-cn"

        language_name = LANGUAGE_NAMES.get(
            detected,
            detected.upper()
        )

        print(
            f"🌍 Detected language: "
            f"{language_name} "
            f"({confidence * 100:.2f}%)"
        )

        return detected

    except Exception as e:

        print(
            f"⚠️ Language detection failed: {e}"
        )

        return "en"


# ============================================================
# MIXED-LANGUAGE DETECTION
# ============================================================

def detect_mixed_language(text):

    words = text.split()

    if len(words) < 8:
        return []

    chunks = []

    chunk_size = 6

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk)

    detected_languages = []

    for chunk in chunks:

        try:

            result = language_detector(
                chunk[:256]
            )[0]

            language = result["label"].lower()

            if language in {
                "zh",
                "zh-cn",
                "zh-tw"
            }:
                language = "zh-cn"

            if language not in detected_languages:

                detected_languages.append(
                    language
                )

        except Exception:

            continue

    return detected_languages


# ============================================================
# SENTIMENT ANALYSIS
# ============================================================

def analyze_sentiment(text):

    print()
    print("🧠 Analyzing sentiment...")

    try:

        result = sentiment_analyzer(
            text[:512]
        )[0]

        label = result["label"].lower()

        confidence = float(
            result["score"]
        )

        if label == "positive":

            emoji = "😊"

            instruction = (
                "The user appears positive. "
                "Respond in a friendly, "
                "encouraging, and positive tone."
            )

        elif label == "negative":

            emoji = "😟"

            instruction = (
                "The user appears frustrated "
                "or negative. Respond empathetically "
                "and calmly. Acknowledge their concern "
                "and avoid sounding dismissive."
            )

        else:

            emoji = "😐"

            instruction = (
                "The user's sentiment appears neutral. "
                "Respond clearly, professionally, "
                "and directly."
            )

        print()
        print("=" * 60)
        print("😊 SENTIMENT ANALYSIS")
        print("-" * 60)

        print(
            f"Sentiment: {emoji} {label}"
        )

        print(
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        return {
            "label": label,
            "confidence": confidence,
            "instruction": instruction
        }

    except Exception as e:

        print(
            f"⚠️ Sentiment analysis failed: {e}"
        )

        return {
            "label": "neutral",
            "confidence": 0.0,
            "instruction": (
                "Respond clearly and professionally."
            )
        }


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

        return (
            "No relevant research papers "
            "were found."
        )

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
# TRANSLATE / NORMALIZE QUERY FOR RETRIEVAL
# ============================================================

def prepare_retrieval_query(
    question,
    language,
    chat_history
):

    language_name = LANGUAGE_NAMES.get(
        language,
        language
    )

    if language == "en":
        return question

    prompt = f"""
You are a multilingual research retrieval assistant.

The user's current language is:
{language_name}

Convert the user's question into a concise
English search query for an English-language
arXiv research database.

Preserve:

- technical terminology
- entities
- intent
- important context
- references to previous conversation

Do NOT answer the question.

Return ONLY the English search query.

Previous conversation:
{chat_history[-3000:]}

Current user question:
{question}
"""

    try:

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        query = response.text.strip()

        if query:
            return query

    except Exception as e:

        print(
            f"⚠️ Cross-lingual query preparation "
            f"failed: {e}"
        )

    return question


# ============================================================
# GENERATE RESEARCH ANSWER
# ============================================================

def generate_research_answer(
    question,
    collection,
    language,
    sentiment,
    chat_history
):

    retrieval_query = prepare_retrieval_query(
        question,
        language,
        chat_history
    )

    results = search_papers(
        collection,
        retrieval_query,
        top_k=TOP_K
    )

    context = build_context(
        results
    )

    language_name = LANGUAGE_NAMES.get(
        language,
        language
    )

    mixed_languages = detect_mixed_language(
        question
    )

    mixed_language_info = ""

    if len(mixed_languages) > 1:

        names = [
            LANGUAGE_NAMES.get(
                lang,
                lang
            )
            for lang in mixed_languages
        ]

        mixed_language_info = (
            "\nThe user's message contains "
            "multiple detected languages: "
            + ", ".join(names)
            + ". Handle the mixed-language "
            "input naturally and preserve its "
            "meaning.\n"
        )

    prompt = f"""
You are a multilingual AI research assistant
specialized in Computer Science.

You are part of a multilingual conversational
RAG system.

CURRENT USER LANGUAGE:
{language_name}

IMPORTANT MULTILINGUAL RULES:

1. Understand the user's message regardless
   of which supported language they use.

2. Preserve the meaning and intent of the
   user's question.

3. The user may switch languages during the
   same conversation.

4. The user may mix multiple languages in
   the same message.

5. Do NOT lose conversational context when
   the user switches languages.

6. Answer primarily in the language used
   by the current user message.

7. If the user explicitly requests another
   language, answer in that requested language.

8. Keep technical names, paper titles,
   arXiv IDs, model names, and mathematical
   notation accurate.

9. If a technical term is normally written
   in English, it is acceptable to keep the
   technical term in English while explaining
   it in the user's language.

10. Resolve references such as:
    "this paper", "that method", "the second one",
    "iska", "eta", "esto", etc. using the
    previous conversation context.

11. If the question is ambiguous, use the
    available conversation context to infer
    the intended meaning.

12. If the ambiguity cannot be resolved,
    ask a short clarification question.

SENTIMENT:

{sentiment["instruction"]}

{mixed_language_info}

RESEARCH ANSWERING RULES:

1. Base research claims primarily on the
   retrieved research context.

2. Do not invent facts, results, authors,
   experiments, or citations.

3. If the retrieved papers do not contain
   enough information, clearly say so.

4. Explain technical concepts clearly.

5. When useful, compare approaches between
   papers.

6. Mention relevant paper titles when making
   research claims.

7. Distinguish evidence from interpretation.

8. Do not claim a paper proves something unless
   the retrieved context supports that claim.

9. Use Markdown headings and bullet points
   when appropriate.

10. If the user is simply greeting you,
    thanking you, or making casual conversation,
    do NOT force an academic research answer.
    Respond naturally.

CONVERSATION HISTORY:
=====================

{chat_history[-8000:]}

RETRIEVED RESEARCH:
===================

{context}

CURRENT USER MESSAGE:
=====================

{question}

Now respond naturally and appropriately.

Preserve conversation continuity and answer
in the current user's language.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
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
    print("🌍 MULTILINGUAL ARXIV RESEARCH ASSISTANT")
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
        "🌍 Supported conversation languages:"
    )

    print(
        "English, Hindi, Bengali, Spanish, "
        "French, German, Italian, Portuguese, "
        "Japanese, Korean, Chinese, Arabic, "
        "Russian, Tamil, Telugu, Marathi, "
        "Gujarati, Punjabi"
    )

    print()

    print(
        "💬 Language can be changed at any time."
    )

    print(
        "💬 Mixed-language messages are supported."
    )

    print(
        "💬 Conversation context is preserved."
    )

    print()

    print(
        "Type 'exit' to quit."
    )

    conversation_history = []

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
                "\n👋 Exiting multilingual "
                "research assistant."
            )

            break

        try:

            # ------------------------------------------------
            # LANGUAGE DETECTION
            # ------------------------------------------------

            print()
            print(
                "🌍 Detecting language..."
            )

            language = detect_language(
                question
            )

            # ------------------------------------------------
            # SENTIMENT ANALYSIS
            # ------------------------------------------------

            sentiment = analyze_sentiment(
                question
            )

            # ------------------------------------------------
            # BUILD HISTORY
            # ------------------------------------------------

            history_text = ""

            for item in conversation_history:

                history_text += (
                    f"User ({item['language']}): "
                    f"{item['user']}\n"
                    f"Assistant: "
                    f"{item['assistant']}\n\n"
                )

            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

            print()
            print(
                "🧠 Retrieving research and "
                "generating multilingual answer..."
            )

            answer, results = (
                generate_research_answer(
                    question,
                    collection,
                    language,
                    sentiment,
                    history_text
                )
            )

            # ------------------------------------------------
            # DISPLAY ANSWER
            # ------------------------------------------------

            print()
            print("=" * 60)
            print("💡 RESEARCH ANSWER")
            print("=" * 60)

            print()
            print(answer)

            # ------------------------------------------------
            # DISPLAY SOURCES
            # ------------------------------------------------

            display_sources(
                results
            )

            # ------------------------------------------------
            # SAVE CONVERSATION MEMORY
            # ------------------------------------------------

            conversation_history.append(
                {
                    "user": question,
                    "assistant": answer,
                    "language": language,
                    "sentiment": sentiment["label"]
                }
            )

            # Keep the latest 10 turns.
            if len(conversation_history) > 10:

                conversation_history = (
                    conversation_history[-10:]
                )

            print()
            print(
                f"🧠 Context memory: "
                f"{len(conversation_history)} "
                f"conversation turn(s)"
            )

        except Exception as e:

            print()
            print(
                "❌ Error generating answer:"
            )

            print(e)

            print()
            print(
                "The conversation memory was "
                "not updated for this failed turn."
            )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()