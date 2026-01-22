
from src.loader import load_pdf
from src.chunker import chunk_documents
from src.embeddings import VectorStore
from src.retriever import ReRanker
from src.llm import LocalLLM

vector_store = VectorStore()
reranker = ReRanker()
llm = LocalLLM()

def build_index(apple_path, tesla_path):
    docs = []
    docs.extend(load_pdf(apple_path, "Apple 10-K"))
    docs.extend(load_pdf(tesla_path, "Tesla 10-K"))
    chunks = chunk_documents(docs)
    vector_store.build(chunks)

def answer_question(query: str) -> dict:
    retrieved = vector_store.search(query)
    reranked = reranker.rerank(query, retrieved)

    if not reranked:
        return {
            "answer": "This question cannot be answered based on the provided documents.",
            "sources": []
        }

    context = "\n\n".join(
        f"[{c['metadata']['document']} p.{c['metadata']['page']}]\n{c['text']}"
        for c in reranked
    )

    prompt = f"""You are a financial compliance assistant.

Answer using ONLY the provided context.

Rules:
- If not found, say: "Not specified in the document."
- If out of scope, say:
  "This question cannot be answered based on the provided documents."

Context:
{context}

Question:
{query}

Answer:
"""

    output = llm.generate(prompt)
    answer = output.split("Answer:")[-1].strip()

    return {
        "answer": answer,
        "sources": [
            f"{c['metadata']['document']}, p. {c['metadata']['page']}"
            for c in reranked
        ]
    }
