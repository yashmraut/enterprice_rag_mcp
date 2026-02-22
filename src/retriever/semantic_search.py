from src.ingestion.load_documents import DocumentLoader
from src.chunking.chunker import TextChunker
from src.embeddings.embedder import Embedder
from src.vector_store.faiss_index import FAISSVectorStore
import numpy as np


class SemanticSearchEngine:
    def __init__(self):
        print("Initializing Semantic Search Engine...")

        # Step 1: Load Documents
        loader = DocumentLoader(data_dir="data")
        self.documents = loader.load_all_documents()
        print(f"Loaded {len(self.documents)} documents")

        # Step 2: Chunk Documents
        chunker = TextChunker(chunk_size=1200, overlap=300)
        self.chunks = chunker.chunk_documents(self.documents)
        print(f"Created {len(self.chunks)} chunks")

        # Step 3: Generate Embeddings
        self.embedder = Embedder()
        self.embeddings = self.embedder.embed_chunks(self.chunks)

        # Step 4: Initialize FAISS Index
        dimension = self.embeddings.shape[1]
        self.vector_store = FAISSVectorStore(dimension)

        # Step 5: Add embeddings to FAISS
        self.vector_store.add_embeddings(self.embeddings, self.chunks)

        print("Semantic Search Engine Ready!")

    def search(self, query: str, top_k):
        print(f"\nSearching for: {query}")

        # Convert query to embedding
        query_embedding = self.embedder.model.encode(query, convert_to_numpy=True)

        # Search FAISS
        results = self.vector_store.search(query_embedding, top_k)

        return results


if __name__ == "__main__":
    engine = SemanticSearchEngine()

    while True:
        query = input("\nEnter your query (or type 'exit'): ")

        if query.lower() == "exit":
            break

        results = engine.search(query, top_k=5)

        print("\nTop Results:\n")
        for i, res in enumerate(results, 1):
            print(f"Rank {i} | Score: {res['score']:.4f}")
            print(f"Source: {res['source']}")
            print(f"Chunk Preview: {res['chunk'][:1200]}")
            print("-" * 80)