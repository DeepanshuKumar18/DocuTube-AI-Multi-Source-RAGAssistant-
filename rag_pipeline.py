import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from config import LLM_MODEL, EMBEDDING_MODEL
from loaders import load_youtube_docs, load_pdf_docs


# vectorstore builde
@st.cache_resource(show_spinner=True)
def _build_vectorstore_from_docs(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(docs), len(chunks)


# Public API: document → vectorstore
def build_vectorstore(youtube_url: str | None, pdf_path: str | None):
    docs = []

    # Load YouTube transcript
    if youtube_url:
        docs.extend(load_youtube_docs(youtube_url))

    # Load PDF documents
    if pdf_path:
        docs.extend(load_pdf_docs(pdf_path))

    if not docs:
        raise ValueError("No valid documents were loaded.")

    # Call function ONLY after validation
    return _build_vectorstore_from_docs(docs)


# Build RAG Chain
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.2,
        timeout=30,
        max_retries=2,
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Answer ONLY from the provided context.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"],
    )

    def combine_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(combine_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
