from pathlib import Path
from typing import List, Dict
from pdfminer.high_level import extract_text

class DocumentLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def clean_text(self, text: str) -> str:
        text = text.replace("\n\n", "\n")
        text = " ".join(text.split())
        return text

    def load_pdf_files(self) -> List[Dict]:
        documents = []

        for file_path in self.data_dir.rglob("*.pdf"):
            try:
                text = extract_text(file_path)
                text = self.clean_text(text)

                if text and text.strip():
                    documents.append({
                        "content": text,
                        "source": str(file_path),
                        "file_name": file_path.name,
                        "file_type": "pdf"
                    })
                else:
                    print(f"Warning: No extractable text in {file_path}")

            except Exception as e:
                print(f"Error loading PDF {file_path}: {e}")

        return documents

    def load_all_documents(self) -> List[Dict]:
        pdf_docs = self.load_pdf_files()
        return pdf_docs
