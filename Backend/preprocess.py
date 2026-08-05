from pypdf import PdfReader
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_PATH = os.path.join(
    BASE_DIR,
    "knowledge",
    "Laws-of-Cricket-2017-Code-3rd-Edition-2022_1.pdf"
)

OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "mcc_chunks.json"
)


def split_text(text, chunk_size=1200, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


def create_chunks():

    reader = PdfReader(PDF_PATH)

    data = []

    idx = 0

    for page_no, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text:
            continue

        page_chunks = split_text(text)

        for chunk in page_chunks:

            data.append({

                "id": idx,

                "page": page_no,

                "text": chunk

            })

            idx += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Created {len(data)} chunks")


if __name__ == "__main__":
    create_chunks()