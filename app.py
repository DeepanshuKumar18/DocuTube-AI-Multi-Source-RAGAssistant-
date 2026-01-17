import os
import streamlit as st
from config import GROQ_API_KEY
from rag_pipeline import build_vectorstore, build_rag_chain

# Configuration
if not GROQ_API_KEY:
    st.error("API key not found. Please set API in your environment.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

st.set_page_config(page_title="DocuTube AI", layout="wide")

st.markdown("🤖 DocuTube AI")
st.caption("Chat with YouTube Videos & PDFs using RAG")

# Session State
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Sidebar
with st.sidebar:
    st.header("Knowledge Source")

    source_type = st.radio("Select source", ["YouTube", "PDF", "Both"])

    youtube_url = None
    pdf_file = None

    if source_type in ["YouTube", "Both"]:
        youtube_url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=xxxx"
        )

    if source_type in ["PDF", "Both"]:
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    st.divider()

    build_btn = st.button("Build Knowledge Base", use_container_width=True)

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# Build Knowledge Base
if build_btn:
    if source_type in ["YouTube", "Both"] and not youtube_url:
        st.error("Please provide a YouTube URL")
        st.stop()

    if source_type in ["PDF", "Both"] and not pdf_file:
        st.error("Please upload a PDF")
        st.stop()

    pdf_path = None
    if pdf_file:
        pdf_path = f"temp_{pdf_file.name}"
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())

    with st.spinner("Processing documents..."):
        try:
            vectorstore, doc_count, chunk_count = build_vectorstore(
                youtube_url=youtube_url,
                pdf_path=pdf_path,
            )
        except ValueError as e:
            # Expected loader errors (PDF unreadable, no transcript, etc.)
            st.error(str(e))
            st.stop()
        except Exception:
            st.error(
                "Failed to process documents."
                "Please try again or use a different source."
            )
            st.stop()

    st.session_state.rag_chain = build_rag_chain(vectorstore)
    st.session_state.chat_history = []

    st.success(
        f"Knowledge base ready " f"({doc_count} documents, {chunk_count} chunks)"
    )


# UI
if st.session_state.rag_chain:

    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])

    user_query = st.chat_input("Ask something about the content...")

    if user_query:
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.spinner("Generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke(user_query)
            except Exception:
                answer = (
                    "**service temporarily unavailable**\n\n"
                    "Please try again shortly."
                )

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.chat_history.append({"question": user_query, "answer": answer})
