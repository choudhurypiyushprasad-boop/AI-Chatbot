from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_FOLDER = Path("data/docs")


def split_pdf():

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    pdfs = list(PDF_FOLDER.glob("*.pdf"))

    for pdf in pdfs:

        print(f"\nReading {pdf.name}")

        reader = PdfReader(pdf)

        text = ""

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text

        chunks = splitter.split_text(text)

        print(f"Total chunks : {len(chunks)}")

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n------ Chunk {i+1} ------\n")
            print(chunk)


if __name__ == "__main__":
    split_pdf()