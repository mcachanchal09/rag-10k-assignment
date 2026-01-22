
from pypdf import PdfReader

def load_pdf(path, doc_name):
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "text": text,
                "metadata": {
                    "document": doc_name,
                    "page": i + 1
                }
            })
    return pages
