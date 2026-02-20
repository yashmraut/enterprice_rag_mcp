from typing import List, Dict

class TextChunker:
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, document: Dict) -> List[Dict]:
        text = document["content"]
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "chunk_content": chunk_text,
                "source": document["source"],
                "file_name": document["file_name"]
            })

            start += self.chunk_size - self.overlap

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        all_chunks = []

        for doc in documents:
            chunks = self.chunk_text(doc)
            all_chunks.extend(chunks)

        return all_chunks


if __name__ == "__main__":
    from src.ingestion.load_documents import DocumentLoader

    loader = DocumentLoader(data_dir="data")
    docs = loader.load_all_documents()

    chunker = TextChunker(chunk_size=500, overlap=100)
    chunks = chunker.chunk_documents(docs)

    print(f"Total chunks created: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:10]):
        print(f"Chunk {i+1}")
        print(f"Source: {chunk['file_name']}")
        print(f"Preview: {chunk['chunk_content'][:500]}\n")
        print("-" * 50)
