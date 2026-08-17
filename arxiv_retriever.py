import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DATA_FILE = Path(
    "arxiv/data/arxiv_cs.json"
)

VECTOR_STORE = "arxiv_vector_store"

COLLECTION_NAME = "arxiv_papers"

TOP_K = 5


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("🧠 Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ==========================================
# LOAD PAPERS
# ==========================================

def load_papers():

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
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


# ==========================================
# CREATE VECTOR DATABASE
# ==========================================

def create_database(papers):

    print("💾 Creating ChromaDB...")

    client = chromadb.PersistentClient(
        path=VECTOR_STORE
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    existing = collection.count()

    if existing > 0:

        print(
            f"ℹ️ Database already contains "
            f"{existing} papers"
        )

        return collection


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


        documents.append(
            document
        )


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
                )
            }
        )


    print(
        "🧠 Creating embeddings..."
    )


    embeddings = model.encode(
        documents,
        show_progress_bar=True
    )


    print(
        "💾 Storing papers in ChromaDB..."
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


# ==========================================
# SEARCH PAPERS
# ==========================================

def search_papers(
    collection,
    query,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [query]
    )[0]


    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )


    return results


# ==========================================
# CONVERT CHROMADB RESULTS
# ==========================================

def normalize_results(
    results
):

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

    ids = results.get(
        "ids",
        [[]]
    )[0]


    papers = []


    for index, document in enumerate(
        documents
    ):

        metadata = {}

        if index < len(metadatas):

            metadata = (
                metadatas[index]
                or {}
            )


        distance = None

        if index < len(distances):

            distance = distances[index]


        paper_id = ""

        if index < len(ids):

            paper_id = ids[index]


        paper = {

            "title": metadata.get(
                "title",
                "Unknown"
            ),

            "authors": metadata.get(
                "authors",
                ""
            ),

            "categories": metadata.get(
                "categories",
                ""
            ),

            "published": metadata.get(
                "published",
                ""
            ),

            "arxiv_id": metadata.get(
                "arxiv_id",
                paper_id
            ),

            "abstract": document,

            "distance": distance
        }


        papers.append(
            paper
        )


    return papers


# ==========================================
# DISPLAY RESULTS
# ==========================================

def display_results(
    results
):

    papers = normalize_results(
        results
    )


    if not papers:

        print(
            "❌ No relevant papers found."
        )

        return


    print()

    print(
        "🔎 RELEVANT RESEARCH PAPERS"
    )

    print("=" * 60)


    for index, paper in enumerate(
        papers,
        start=1
    ):

        print()

        print(
            f"RESULT {index}"
        )

        print("-" * 60)


        print(
            f"📄 Title: "
            f"{paper['title']}"
        )


        print(
            f"👨‍🔬 Authors: "
            f"{paper['authors']}"
        )


        print(
            f"🏷️ Categories: "
            f"{paper['categories']}"
        )


        print(
            f"📅 Published: "
            f"{paper['published']}"
        )


        print(
            f"🔗 arXiv: "
            f"{paper['arxiv_id']}"
        )


        print(
            f"📏 Distance: "
            f"{paper['distance']}"
        )


        print()

        print(
            "📝 Abstract:"
        )

        print(
            paper["abstract"]
        )