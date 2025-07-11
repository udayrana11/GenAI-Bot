import os
import streamlit as st

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ─── Embeddings via Hugging Face Inference API ──────────────────────────────────
hf_token = st.secrets["huggingface"]["token"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# ─── Streamlit UI ───────────────────────────────────────────────────────────────
st.title("Conversational RAG With PDF Uploads & Chat History")
st.write("Upload PDFs and chat with their content.")

api_key = st.text_input("Enter your Groq API key:", type="password")
if not api_key:
    st.warning("Please enter your Groq API Key above to begin.")
    st.stop()

llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

session_id = st.text_input("Session ID", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader(
    "Choose PDF file(s)", type="pdf", accept_multiple_files=True
)

if uploaded_files:
    # Load & split all uploaded PDFs
    documents = []
    for uploaded in uploaded_files:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded.getvalue())
        loader = PyPDFLoader("temp.pdf")
        documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000, chunk_overlap=500
    )
    splits = text_splitter.split_documents(documents)

    # Build vectorstore & retriever
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings
    )
    retriever = vectorstore.as_retriever()

    # History-aware retriever setup
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question, "
                "formulate a standalone question understandable on its own.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # QA chain
    qa_system = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant for question-answering tasks. "
                "Use retrieved context to answer concisely (≤3 sentences). "
                "{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm, qa_system)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Wrap with message history
    def get_session_history(sid: str) -> BaseChatMessageHistory:
        if sid not in st.session_state.store:
            st.session_state.store[sid] = ChatMessageHistory()
        return st.session_state.store[sid]

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # User question input
    user_input = st.text_input("Your question:")
    if user_input:
        history = get_session_history(session_id)
        result = conversational_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.markdown("**Assistant:** " + result["answer"])
        st.markdown("**Chat History:**")
        for msg in history.messages:
            sender = msg["type"]
            content = msg["data"]["content"]
            st.write(f"- *{sender}*: {content}")
