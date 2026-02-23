import streamlit as st
from src.rag.groq_llm_rag_pipeline import GROQ_RAGPipeline

# Page config
st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("Enterprise Knowledge Assistant (RAG + Groq)")
st.markdown("Ask questions based on your documents using Retrieval-Augmented Generation.")

# Cache the RAG pipeline (VERY IMPORTANT for performance)
@st.cache_resource
def load_rag_pipeline():
    return GROQ_RAGPipeline()

# Initialize pipeline
rag = load_rag_pipeline()

# Sidebar info
with st.sidebar:
    st.header("⚙️ System Info")
    st.write("Model: llama-3.3-70b-versatile (Groq)")
    st.write("Retriever: FAISS + MiniLM Embeddings")
    st.write("Mode: Grounded RAG")
    st.markdown("---")
    st.write("Tip: Ask questions related to your uploaded PDFs.")

# Chat input
query = st.text_input(" Ask a question about your documents:")

# Top-K selector (advanced control)
top_k = st.slider("Number of retrieved chunks (Top-K)", 1, 10, 5)

if st.button("🔍 Get Answer") and query:
    with st.spinner("Retrieving context and generating answer..."):
        try:
            answer, sources = rag.ask(query, top_k=top_k)

            # Display Answer
            st.subheader(" Answer")
            st.write(answer)

            # Display Sources (Transparency = BIG plus in interviews)
            st.subheader("Retrieved Sources (RAG Transparency)")
            
            if sources:
                for i, src in enumerate(sources, 1):
                    with st.expander(f"Source {i} | Score: {src['score']:.4f} | File: {src['source']}"):
                        st.write(src['chunk'])
            else:
                st.warning("No relevant sources found.")

        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.caption("Built with FAISS + SentenceTransformers + Groq LLM + Streamlit")