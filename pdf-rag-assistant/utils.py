from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_pdf_text(pdf_files):
    text = ""

    for pdf in pdf_files:
        reader = PdfReader(pdf)

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


def create_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return splitter.split_text(text)
