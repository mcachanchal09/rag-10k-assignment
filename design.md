
## Design Overview

- Chunking: 1000 chars with 200 overlap

- Embeddings: all-MiniLM-L6-v2 (It provides strong semantic performance for short and medium-length text and It is lightweight and fast, making it suitable for Colab environments)

- Vector DB: FAISS (High-performance similarity search)

- Re-ranking: Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)

- LLM: Phi-3 Mini (It is lightweight and runs efficiently in Colab)

Designed for accuracy, reproducibility, and zero hallucination.

Out-of-Scope Handling (Zero Hallucination)

To prevent hallucinations, the system enforces a strict rule:

If no retrieved chunk exceeds a relevance threshold, the system returns:

“This question cannot be answered based on the provided documents.”