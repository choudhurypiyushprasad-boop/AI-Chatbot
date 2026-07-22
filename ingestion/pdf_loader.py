print("Hello - SCRIPT STARTED")


from pathlib import Path
from pypdf import PdfReader

PDF_FOLDER = Path("data/docs")


def load_pdfs():
    pdf_files = list(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in data/docs")
        return

    for pdf in pdf_files:
        print(f"\n📄 Reading: {pdf.name}")

        reader = PdfReader(pdf)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text

        print("=" * 60)
        print(text[:500])      # Show first 500 characters
        print("=" * 60)


if __name__ == "__main__":
    load_pdfs()