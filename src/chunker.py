
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []
    for doc in documents:
        for chunk in splitter.split_text(doc["text"]):
            chunks.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })
    return chunks
