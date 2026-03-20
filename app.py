import os
import streamlit as st
import asyncio
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

# --- 1. CONFIGURATION ---
# Task 2.4: Model Disclosure
os.environ["GROQ_API_KEY"] = "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" # <--- PASTE YOUR KEY HERE
os.environ["HF_TOKEN"] = "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" # Optional
RETRIEVER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL = "llama-3.1-8b-instant"

st.set_page_config(page_title="Chapter 7 AI Assistant", layout="centered")
st.title("🤖 Chapter 7: Contextual RAG Chatbot")
st.caption(f"Retriever: {RETRIEVER_MODEL} | Generator: {GENERATOR_MODEL}")

# --- 2. INITIALIZE MODELS ---
@st.cache_resource
def init_models():
    llm = ChatGroq(model_name=GENERATOR_MODEL, temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name=RETRIEVER_MODEL)
    return llm, embeddings

llm, embeddings = init_models()

# --- 3. CONTEXTUAL RETRIEVAL LOGIC (Task 2.2) ---
async def enrich_chunk(content, full_text):
    """Adds a situational prefix to chunks to improve retrieval accuracy"""
    prompt = f"Provide a 1-sentence context for this chunk within Chapter 7 (LLMs): {content[:300]}"
    try:
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        return f"Context: {res.content.strip()}\n\n{content}"
    except:
        return content

@st.cache_resource
def build_index():
    if not os.path.exists("7.pdf"):
        st.error("Please upload '7.pdf' to the Colab sidebar!")
        st.stop()
    
    loader = PyPDFLoader("7.pdf")
    docs = loader.load()
    full_text = " ".join([d.page_content for d in docs])
    
    # Task 2.1: Chunking strategy
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
    chunks = splitter.split_documents(docs)
    
    enriched_docs = []
    with st.status("🛠️ Creating Contextual Embeddings...", expanded=True) as status:
        prog = st.progress(0)
        
        async def process():
            for i, chunk in enumerate(chunks):
                txt = await enrich_chunk(chunk.page_content, full_text)
                enriched_docs.append(Document(page_content=txt, metadata=chunk.metadata))
                prog.progress((i + 1) / len(chunks))
                await asyncio.sleep(0.05)
        
        asyncio.run(process())
        status.update(label="✅ Vector Database Ready!", state="complete")
        
    return Chroma.from_documents(enriched_docs, embeddings)

# --- 4. CHAT INTERFACE ---
vdb = build_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if query := st.chat_input("Ask about Large Language Models..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        # Task 3.3: Retrieve relevant chunks
        retriever = vdb.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)
        
        # Build prompt with context
        context_text = "\n\n---\n\n".join([d.page_content for d in docs])
        full_prompt = f"Use the following context to answer the question.\n\nContext:\n{context_text}\n\nQuestion: {query}"
        
        # Generate Answer
        response = llm.invoke(full_prompt)
        st.markdown(response.content)
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        
        # Task 3.4: Display Citations
        with st.expander("📚 View Source Chunks & Metadata"):
            for i, doc in enumerate(docs):
                page_num = doc.metadata.get('page', 'Unknown')
                st.write(f"**Source {i+1} (Page {page_num+1 if isinstance(page_num, int) else page_num})**")
                st.caption(doc.page_content)
