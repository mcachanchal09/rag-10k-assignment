
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class VectorStore:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        embeddings = self.embedder.encode([c["text"] for c in chunks], show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(np.array(embeddings))

    def search(self, query, k=5):
        q_emb = self.embedder.encode([query])
        _, idxs = self.index.search(q_emb, k)
        return [self.chunks[i] for i in idxs[0]]
