import os
import xml.etree.ElementTree as ET
import chromadb
from sentence_transformers import SentenceTransformer

# ==============================
# CONFIGURATION
# ==============================

DATASET_PATH = "MedQuAD"
VECTOR_DB_PATH = "medical_vector_store"
COLLECTION_NAME = "medquad"

# ==============================
# LOAD MODELS
# ==============================

print("🧠 Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# CONNECT TO CHROMADB
# ==============================

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

# Recreate collection to avoid duplicate records
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.create_collection(
    name=COLLECTION_NAME
)

# ==============================
# PARSE XML
# ==============================

def parse_xml(file_path):

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        focus = ""
        question = ""
        answers = []

        # Medical focus
        focus_element = root.find(".//Focus")

        if focus_element is not None:
            focus = "".join(focus_element.itertext()).strip()

        # Question
        question_element = root.find(".//Question")

        if question_element is not None:
            question = "".join(
                question_element.itertext()
            ).strip()

        # Answers
        for answer_element in root.findall(".//Answer"):

            answer = "".join(
                answer_element.itertext()
            ).strip()

            if answer:
                answers.append(answer)

        if not question or not answers:
            return None

        return {
            "focus": focus,
            "question": question,
            "answers": answers
        }

    except Exception as e:

        print(f"⚠️ Error reading {file_path}: {e}")

        return None


# ==============================
# COLLECT RECORDS
# ==============================

records = []

print("=" * 60)
print("🏥 MEDQUAD MEDICAL QA INGESTION")
print("=" * 60)

print("\n🔍 Scanning MedQuAD dataset...")

xml_files = []

for root, dirs, files in os.walk(DATASET_PATH):

    for file in files:

        if file.lower().endswith(".xml"):

            xml_files.append(
                os.path.join(root, file)
            )

print(f"📄 Found {len(xml_files)} XML files")


# ==============================
# PROCESS XML FILES
# ==============================

for index, file_path in enumerate(xml_files):

    record = parse_xml(file_path)

    if record is None:
        continue

    for answer in record["answers"]:

        text = f"""
Medical Topic:
{record['focus']}

Question:
{record['question']}

Answer:
{answer}
""".strip()

        records.append({
            "text": text,
            "focus": record["focus"],
            "question": record["question"],
            "source": file_path
        })

    # Progress
    if (index + 1) % 500 == 0:

        print(
            f"⏳ Processed {index + 1}/{len(xml_files)} files..."
        )


print(f"\n✅ Created {len(records)} medical QA records")


# ==============================
# EMBEDDINGS
# ==============================

print("\n🧠 Creating embeddings...")

texts = [
    record["text"]
    for record in records
]

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
).tolist()


# ==============================
# STORE IN CHROMADB
# ==============================

print("\n💾 Storing records in ChromaDB...")

batch_size = 5000

for start in range(0, len(records), batch_size):

    end = min(
        start + batch_size,
        len(records)
    )

    batch_records = records[start:end]

    collection.add(

        ids=[
            f"medquad_{i}"
            for i in range(start, end)
        ],

        documents=[
            r["text"]
            for r in batch_records
        ],

        embeddings=embeddings[start:end],

        metadatas=[
            {
                "focus": r["focus"],
                "question": r["question"],
                "source": r["source"]
            }

            for r in batch_records
        ]
    )

    print(
        f"✅ Stored {end}/{len(records)} records"
    )


# ==============================
# SUMMARY
# ==============================

print("\n" + "=" * 60)
print("🎉 MEDQUAD INGESTION COMPLETE")
print("=" * 60)

print(
    f"📊 Records in database: {collection.count()}"
)

print(
    f"📁 Vector database: {VECTOR_DB_PATH}"
)

print(
    f"🗂️ Collection: {COLLECTION_NAME}"
)