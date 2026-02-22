from sentence_transformers import SentenceTransformer
from typing import List, Dict
import numpy as np

class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print("loading embedding model")
        self.model = SentenceTransformer(model_name)
        print("embedding model loaded successfully")

    def embed_chunks(self, chunks: List[Dict]) -> np.ndarray:
        texts = [chunk["chunk_content"] for chunk in chunks]

        print(f"generating embeddings for {len(texts)} chunks")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings
        
if __name__ == "__main__":
    from src.ingestion.load_documents import DocumentLoader
    from src.chunking.chunker import TextChunker

    loader = DocumentLoader(data_dir="data")
    docs = loader.load_all_documents()

    chunker =  TextChunker(chunk_size=300, overlap=50)
    chunks = chunker.chunk_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    embedder = Embedder()
    embeddings = embedder.embed_chunks(chunks)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print("sample embedding vector (first five values):")
    print(embeddings[0][:5])    
        