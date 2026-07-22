from pathlib import Path
import chromadb
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

print("✅ Script Started")

PDF_FOLDER = Path("data/docs")

client = chromadb.PersistentClient(path="vector_store")
collection = client.get_or_create_collection(name="documents")

print("✅ Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model Loaded")


def create_embeddings():

    print("Searching for PDFs...")

    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    print("PDF files found:", pdf_files)

    if not pdf_files:
        print("❌ No PDFs found!")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for pdf in pdf_files:

        print(f"\n📄 Processing: {pdf.name}")

        reader = PdfReader(pdf)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        print("Characters extracted:", len(text))

        chunks = splitter.split_text(text)

        print("Chunks:", len(chunks))

        for i, chunk in enumerate(chunks):

            print(f"Embedding chunk {i+1}")

            embedding = model.encode(chunk).tolist()

            collection.add(
                ids=[f"{pdf.stem}_{i}"],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": pdf.name}]
            )

        print("✅ Stored successfully")


if __name__ == "__main__":
    create_embeddings()