import json
import time
from pathlib import Path

import requests


# ==========================================
# CONFIGURATION
# ==========================================

OUTPUT_FILE = Path("arxiv/data/arxiv_cs.json")

MAX_RESULTS = 500

CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.CV",
    "cs.IR",
]


# ==========================================
# ARXIV API
# ==========================================

def build_query():
    """Build a Computer Science category query."""

    return " OR ".join(
        f"cat:{category}"
        for category in CATEGORIES
    )


def fetch_papers():

    query = build_query()

    url = "https://export.arxiv.org/api/query"

    params = {
        "search_query": query,
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "AI-Chatbot-Research-Project/1.0"
    }

    print("=" * 60)
    print("📚 ARXIV COMPUTER SCIENCE DATA INGESTION")
    print("=" * 60)

    print()
    print("🔎 Searching arXiv...")
    print(f"📊 Maximum papers: {MAX_RESULTS}")
    print(f"🏷️ Categories: {', '.join(CATEGORIES)}")
    print()

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    return response.text


# ==========================================
# PARSE ARXIV XML
# ==========================================

def parse_papers(xml_text):

    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_text)

    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):

        title = entry.findtext(
            "atom:title",
            "",
            namespace
        ).strip()

        summary = entry.findtext(
            "atom:summary",
            "",
            namespace
        ).strip()

        paper_id = entry.findtext(
            "atom:id",
            "",
            namespace
        ).strip()

        published = entry.findtext(
            "atom:published",
            "",
            namespace
        ).strip()

        updated = entry.findtext(
            "atom:updated",
            "",
            namespace
        ).strip()

        authors = []

        for author in entry.findall(
            "atom:author",
            namespace
        ):

            name = author.findtext(
                "atom:name",
                "",
                namespace
            ).strip()

            if name:
                authors.append(name)

        categories = []

        for category in entry.findall(
            "atom:category",
            namespace
        ):

            term = category.attrib.get(
                "term"
            )

            if term:
                categories.append(term)

        papers.append(
            {
                "id": paper_id,
                "title": title,
                "abstract": summary,
                "authors": authors,
                "categories": categories,
                "published": published,
                "updated": updated,
            }
        )

    return papers


# ==========================================
# SAVE DATA
# ==========================================

def save_papers(papers):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            papers,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(f"💾 Saved {len(papers)} papers")
    print(f"📁 File: {OUTPUT_FILE}")


# ==========================================
# MAIN
# ==========================================

def main():

    try:

        xml_text = fetch_papers()

        print("🧠 Parsing paper metadata...")

        papers = parse_papers(xml_text)

        if not papers:

            print("❌ No papers were returned.")
            return

        save_papers(papers)

        print()
        print("=" * 60)
        print("🎉 ARXIV INGESTION COMPLETE")
        print("=" * 60)

        print(f"📊 Papers collected: {len(papers)}")

    except requests.RequestException as e:

        print()
        print("❌ Network/API error:")
        print(e)

    except Exception as e:

        print()
        print("❌ Unexpected error:")
        print(e)


if __name__ == "__main__":
    main()