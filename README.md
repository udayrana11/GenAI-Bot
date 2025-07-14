# GenAI-Bot: https://genai-bot-axhicp8o8qk7ghs5wtd5nl.streamlit.app/

Conversational PDF assistant using RAG, Chroma &amp; Groq LLM

# 📚 Conversational RAG Q&A Chatbot with PDF & Chat History

An end-to-end Retrieval-Augmented Generation (RAG) demo: upload PDF(s), vectorize their content, and chat with both the document and your own conversation history using a Streamlit UI.

---

## 🔍 Features

1. **PDF Upload**  
   - Drag & drop or browse for one or more PDF files.  
   - Each file is saved temporarily on disk for processing.

2. **Chunking & Embeddings**  
   - Splits text into 5 000-character chunks (500 overlap).  
   - Embeds chunks with HuggingFace’s **all-MiniLM-L6-v2**.

3. **Vector Store**  
   - Stores embeddings in **Chroma** for fast similarity search.

4. **History-Aware Retrieval**  
   - Reformulates follow-up questions into standalone queries.  
   - Uses your chat history to preserve context.

5. **LLM Backend**  
   - Uses **Groq’s** Gemma2-9B model via `langchain-groq`.  
   - Easily swap to OpenAI, Anthropic, etc.

6. **Session Isolation**  
   - Assign your own Session ID to isolate multiple users.  
   - All chat history stored per session.

---
Author: Uday

