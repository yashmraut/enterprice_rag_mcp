import requests
from src.retriever.semantic_search import SemanticSearchEngine

class RAGPipeline:
    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.search_engine = SemanticSearchEngine()

        # Your internal LLM endpoint
        self.llm_url = "http://10.169.194.4:8000/v1/chat/completions"
        self.model_name = "local-model"  # Placeholder (can be changed later)

    def build_context(self, retrieved_chunks):
        """
        Combine top chunks into a single context block
        """
        context = "\n\n".join(
            [f"Source: {chunk['source']}\nContent: {chunk['chunk']}" 
             for chunk in retrieved_chunks]
        )
        return context

    def build_prompt(self, query, context):
        """
        Grounded RAG prompt (VERY IMPORTANT)
        """
        prompt = f"""
You are an AI assistant that answers ONLY using the provided context.
Do NOT use outside knowledge.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""
        return prompt

    def call_llm(self, prompt):
        """
        Call your local hosted LLM (OpenAI-compatible API)
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(self.llm_url, json=payload, headers=headers)

        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            raise Exception(f"LLM API Error: {response.text}")

    def ask(self, query, top_k=3):
        """
        Full RAG flow:
        Query → Retrieval → Context → LLM → Answer
        """
        print(f"\nUser Query: {query}")

        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.search_engine.search(query, top_k=top_k)

        # Step 2: Build context
        context = self.build_context(retrieved_chunks)

        # Step 3: Build grounded prompt
        prompt = self.build_prompt(query, context)

        # Step 4: Get LLM response
        answer = self.call_llm(prompt)

        return answer, retrieved_chunks


if __name__ == "__main__":
    rag = RAGPipeline()

    while True:
        query = input("\nEnter your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        answer, sources = rag.ask(query)

        print("\n===== FINAL ANSWER =====")
        print(answer)

        print("\n===== SOURCES USED (RAG Transparency) =====")
        for i, src in enumerate(sources, 1):
            print(f"{i}. {src['source']} (Score: {src['score']:.4f})")