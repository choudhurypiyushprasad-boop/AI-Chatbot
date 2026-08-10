import hashlib
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# CONFIGURATION
# ==========================================

DOCS_FOLDER = Path("data/docs")
VECTOR_DB_PATH = "vector_store"
COLLECTION_NAME = "documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ==========================================
# LOAD EMBEDDING MODEL
# ==========================================

print("🧠 Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# ==========================================
# CONNECT TO CHROMADB
# ==========================================

client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


# ==========================================
# FILE HASH
# ==========================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            data = file.read(8192)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


# ==========================================
# EXTRACT PDF TEXT
# ==========================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# ==========================================
# SPLIT TEXT
# ==========================================

def split_text(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:

            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ==========================================
# CREATE CHUNK ID
# ==========================================

def create_id(filename, file_hash, chunk_number):

    value = f"{filename}_{file_hash}_{chunk_number}"

    return hashlib.md5(
        value.encode("utf-8")
    ).hexdigest()


# ==========================================
# CHECK IF FILE IS ALREADY INDEXED
# ==========================================

def is_already_indexed(filename, file_hash):

    results = collection.get(
        where={
            "source": filename
        }
    )

    metadata = results.get("metadatas", [])

    for item in metadata:

        if item.get("file_hash") == file_hash:

            return True

    return False


# ==========================================
# REMOVE OLD VERSION
# ==========================================

def remove_old_version(filename):

    results = collection.get(
        where={
            "source": filename
        }
    )

    ids = results.get("ids", [])

    if ids:

        collection.delete(ids=ids)

        print(
            f"🗑️ Removed old version of {filename}"
        )


# ==========================================
# PROCESS PDF
# ==========================================

def process_pdf(pdf_path):

    filename = pdf_path.name

    file_hash = get_file_hash(pdf_path)

    print(f"\n📄 Checking: {filename}")

    # --------------------------------------
    # CHECK IF ALREADY INDEXED
    # --------------------------------------

    if is_already_indexed(filename, file_hash):

        print("✅ Already indexed — skipping")

        return

    # --------------------------------------
    # NEW OR MODIFIED FILE
    # --------------------------------------

    print("🆕 New or modified document detected")

    # Remove previous version if it exists
    remove_old_version(filename)

    # Extract text
    text = extract_pdf_text(pdf_path)

    if not text.strip():

        print("⚠️ No text found — skipping")

        return

    # Split into chunks
    chunks = split_text(text)

    print(
        f"✂️ Created {len(chunks)} chunks"
    )

    # Generate embeddings
    embeddings = embedding_model.encode(
        chunks,
        show_progress_bar=True
    ).tolist()

    ids = []

    documents = []

    metadatas = []

    for i, chunk in enumerate(chunks):

        chunk_id = create_id(
            filename,
            file_hash,
            i
        )

        ids.append(chunk_id)

        documents.append(chunk)

        metadatas.append({
            "source": filename,
            "file_hash": file_hash,
            "chunk": i
        })

    # Add to ChromaDB
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"✅ Added {len(chunks)} chunks to ChromaDB"
    )


# ==========================================
# REMOVE DELETED DOCUMENTS
# ==========================================

def remove_deleted_documents(pdf_files):

    current_files = {
        pdf.name for pdf in pdf_files
    }

    results = collection.get()

    ids_to_delete = []

    for doc_id, metadata in zip(
        results["ids"],
        results["metadatas"]
    ):

        source = metadata.get("source")

        if source not in current_files:

            ids_to_delete.append(doc_id)

    if ids_to_delete:

        collection.delete(
            ids=ids_to_delete
        )

        print(
            f"🗑️ Removed {len(ids_to_delete)} "
            "chunks from deleted documents"
        )


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)

    print("📚 AUTOMATIC KNOWLEDGE BASE UPDATE")

    print("=" * 60)

    DOCS_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_files = list(
        DOCS_FOLDER.glob("*.pdf")
    )

    if not pdf_files:

        print("\n⚠️ No PDF files found")

        return

    print(
        f"\n🔍 Found {len(pdf_files)} PDF file(s)"
    )

    # Process documents
    for pdf in pdf_files:

        try:

            process_pdf(pdf)

        except Exception as e:

            print(
                f"❌ Error processing "
                f"{pdf.name}: {e}"
            )

    # Remove deleted PDFs
    remove_deleted_documents(pdf_files)

    print("\n" + "=" * 60)

    print("🎉 KNOWLEDGE BASE UPDATE COMPLETE")

    print("=" * 60)

    print(
        f"📊 Total chunks: "
        f"{collection.count()}"
    )


# ==========================================
# RUN
# ==========================================

if __name__ == "__main__":

    main()