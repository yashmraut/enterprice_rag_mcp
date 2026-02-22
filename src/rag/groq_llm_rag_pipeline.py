import os
from dotenv import load_dotenv
from groq import Groq
from src.retriever.semantic_search import SemanticSearchEngine

# Load environment variables
load_dotenv()

class GROQ_RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline with Groq LLM...")

        # Initialize retriever (FAISS + embeddings)
        self.search_engine = SemanticSearchEngine()

        # Initialize Groq client
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

        print("RAG Pipeline Ready (Groq LLM Connected)")

    def build_context(self, retrieved_chunks):
        """
        Combine retrieved chunks into a clean context block
        """
        context_blocks = []

        for i, chunk in enumerate(retrieved_chunks, 1):
            block = f"""
[Source {i}: {chunk['source']}]
{chunk['chunk']}
"""
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def build_prompt(self, query: str, context: str) -> str:
        """
        Grounded RAG prompt (anti-hallucination design)
        """
        prompt = f"""
You are an expert AI assistant.

You MUST answer the question ONLY using the provided context.
Do NOT use external knowledge.
If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{query}

Provide a clear, factual, and concise answer based strictly on the context.
"""
        return prompt

    def ask(self, query: str, top_k: int = 5):
        """
        Full RAG flow:
        Query → Retrieval → Context → LLM → Answer
        """
        print(f"\nUser Query: {query}")

        # Step 1: Retrieve relevant chunks (FAISS)
        retrieved_chunks = self.search_engine.search(query, top_k=top_k)

        if not retrieved_chunks:
            return "No relevant context found.", []

        # Step 2: Build context
        context = self.build_context(retrieved_chunks)

        # Step 3: Build grounded prompt
        prompt = self.build_prompt(query, context)

        # Step 4: Call Groq LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a grounded RAG assistant that only answers from provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2  # low = more factual RAG behavior
            # max_tokens=512
        )

        answer = response.choices[0].message.content

        return answer, retrieved_chunks


if __name__ == "__main__":
    rag = GROQ_RAGPipeline()

    print("\n=== RAG System Ready (Groq + FAISS) ===")

    while True:
        query = input("\nEnter your question (type 'exit' to quit): ")

        if query.lower() == "exit":
            break

        answer, sources = rag.ask(query, top_k=5)

        print("\n===== FINAL RAG ANSWER =====\n")
        print(answer)

        print("\n===== RETRIEVED SOURCES (Transparency) =====")
        for i, src in enumerate(sources, 1):
            print(f"{i}. File: {src['source']} | Score: {src['score']:.4f}")