import faiss
import numpy as np
from typing import List, Dict


class FAISSVectorStore:
    def __init__(self, dimension: int):
        """
        dimension = embedding vector size (384 for MiniLM)
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (for cosine similarity)
        self.chunks = []  # Stores metadata + text

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity search
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def add_embeddings(self, embeddings: np.ndarray, chunks: List[Dict]):
        """
        Add embeddings + corresponding chunks into FAISS index
        """
        if len(embeddings) != len(chunks):
            raise ValueError("Embeddings count must match chunks count")

        # Normalize embeddings (VERY IMPORTANT)
        normalized_embeddings = self._normalize_vectors(embeddings)

        # Convert to float32 (FAISS requirement)
        normalized_embeddings = normalized_embeddings.astype("float32")

        # Add to index
        self.index.add(normalized_embeddings)

        # Store chunks for retrieval mapping
        self.chunks.extend(chunks)

        print(f"Added {len(embeddings)} embeddings to FAISS index.")
        print(f"Total vectors in index: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Perform semantic search
        """
        # Normalize query embedding
        query_embedding = self._normalize_vectors(query_embedding.reshape(1, -1))
        query_embedding = query_embedding.astype("float32")

        # Search FAISS index
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "score": float(score),
                    "chunk": self.chunks[idx]["chunk_content"],
                    "source": self.chunks[idx]["file_name"]
                })

        return results