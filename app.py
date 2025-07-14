import streamlit as st
import os
from dotenv import load_dotenv

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ✅ Load environment variables
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

# ✅ Page Setup
st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("🧠 Conversational RAG with PDF + Chat History")
st.info("🔑 Enter your Groq API key and upload PDFs to begin.")
st.secrets["HF_TOKEN"]

# ✅ API key input
api_key = st.text_input("Enter your Groq API Key:", type="password")

# ✅ Upload PDFs
uploaded_files = st.file_uploader("Upload one or more PDF files:", type="pdf", accept_multiple_files=True)

# ✅ Session ID input
session_id = st.text_input("Enter Session ID:", value="default_session")

# ✅ Proceed if both API key and files are present
if api_key and uploaded_files:
    try:
        llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        documents = []
        for uploaded_file in uploaded_files:
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            documents.extend(docs)

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = splitter.split_documents(documents)

        with st.spinner("🔄 Generating vector embeddings..."):
            vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()


        # Build prompts
        contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", "Reformulate the user question to be standalone based on chat history."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the context below to answer. Be concise.\n\n{context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state:
                st.session_state[session] = ChatMessageHistory()
            return st.session_state[session]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        # ✅ Input box for user question
        user_input = st.text_input("Ask your question:")
        if user_input:
            history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            st.write("🤖 Assistant:", response['answer'])
            st.write("🧾 Chat History:", history.messages)

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
else:
    st.warning("Please enter your Groq API key and upload at least one PDF to continue.")
