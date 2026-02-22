from typing import List, Dict
import re


class TextChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 300):
        """
        chunk_size: Maximum characters per chunk (recommended 800–1200 for PDFs)
        overlap: Overlapping characters between chunks to preserve context
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into logical paragraphs.
        This works much better than raw character slicing for PDFs.
        """
        # Split on double newlines or large spacing (common in PDFs)
        paragraphs = re.split(r'\n\s*\n', text)

        # Clean empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        return paragraphs

    def _clean_paragraph(self, paragraph: str) -> str:
        """
        Additional cleaning for PDF-extracted text (pdfminer noise reduction)
        """
        # Remove excessive whitespace
        paragraph = re.sub(r'\s+', ' ', paragraph)

        # Remove standalone page numbers (common in PDFs)
        paragraph = re.sub(r'\bPage\s*\d+\b', '', paragraph, flags=re.IGNORECASE)

        return paragraph.strip()

    def chunk_text(self, document: Dict) -> List[Dict]:
        """
        Convert one document into semantic chunks
        """
        text = document["content"]
        paragraphs = self._split_into_paragraphs(text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = self._clean_paragraph(para)

            # If paragraph alone is bigger than chunk_size, split it
            if len(para) > self.chunk_size:
                for i in range(0, len(para), self.chunk_size):
                    sub_chunk = para[i:i + self.chunk_size]
                    chunks.append({
                        "chunk_content": sub_chunk,
                        "source": document["source"],
                        "file_name": document["file_name"]
                    })
                continue

            # Build chunk logically
            if len(current_chunk) + len(para) <= self.chunk_size:
                current_chunk += " " + para
            else:
                # Save current chunk
                chunks.append({
                    "chunk_content": current_chunk.strip(),
                    "source": document["source"],
                    "file_name": document["file_name"]
                })

                # Start new chunk with overlap logic
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para

        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "chunk_content": current_chunk.strip(),
                "source": document["source"],
                "file_name": document["file_name"]
            })

        return chunks

    def chunk_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Chunk all documents into a single chunk list
        """
        all_chunks = []

        for doc in documents:
            doc_chunks = self.chunk_text(doc)
            all_chunks.extend(doc_chunks)

        print(f"Total chunks created: {len(all_chunks)}")
        return all_chunks


if __name__ == "__main__":
    from src.ingestion.load_documents import DocumentLoader

    # Load documents (PDF + others)
    loader = DocumentLoader(data_dir="data")
    documents = loader.load_all_documents()

    print(f"Total documents loaded: {len(documents)}")

    # Recommended config for PDFs
    chunker = TextChunker(chunk_size=1200, overlap=300)

    chunks = chunker.chunk_documents(documents)

    print("\nSample Chunks Preview:\n")
    for i, chunk in enumerate(chunks[:3], 1):
        print(f"Chunk {i}")
        print(f"Source: {chunk['file_name']}")
        print(f"Length: {len(chunk['chunk_content'])} characters")
        print(f"Preview: {chunk['chunk_content'][:1200]}")
        print("-" * 80)
