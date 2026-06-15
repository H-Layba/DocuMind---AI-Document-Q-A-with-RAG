import os
import tempfile

# Prevent `transformers` from trying to import TensorFlow/Flax (avoids
# Keras 3 incompatibility errors). We only need the PyTorch backend.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# ----------------------------------------------------------------------
# PAGE CONFIG + STYLING
# ----------------------------------------------------------------------
st.set_page_config(page_title="DocuMind | AI Document Q&A", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #0f1117;}
    .stApp {background: linear-gradient(135deg, #1a1c2e 0%, #16213e 100%);}
    h1 {
        background: linear-gradient(90deg, #f72585, #7209b7, #4361ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .stChatMessage {border-radius: 16px;}
    .source-box {
        background-color: rgba(114, 9, 183, 0.12);
        border-left: 4px solid #7209b7;
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #f72585, #7209b7);
        color: white;
        border-radius: 10px;
        border: none;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 DocuMind — Chat With Your Documents")
st.caption("Upload PDFs, ask questions, get answers with cited sources — powered by RAG + LangChain + Groq")

# ----------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------------------------------------------------
# SIDEBAR — SETUP
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Get a free key at https://console.groq.com/keys",
    )

    model_name = st.selectbox(
        "Model",
        options=["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-120b"],
        index=0,
        help="Faster models = quicker answers. 70b is smarter but slower.",
    )

    uploaded_files = st.file_uploader(
        "Upload PDF document(s)", type=["pdf"], accept_multiple_files=True
    )

    chunk_size = st.slider("Chunk size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, step=50)
    top_k = st.slider("Chunks to retrieve (k)", 1, 8, 3)

    process_btn = st.button("🚀 Process Documents", use_container_width=True)

    if st.session_state.vectorstore is not None:
        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.messages = []
            st.rerun()

# ----------------------------------------------------------------------
# DOCUMENT PROCESSING
# ----------------------------------------------------------------------
if process_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    elif not groq_api_key:
        st.sidebar.error("Please enter your Groq API key.")
    else:
        with st.spinner("Reading and indexing your documents... this may take a minute"):
            all_docs = []
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name
                try:
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = f.name
                    all_docs.extend(docs)
                finally:
                    os.unlink(tmp_path)

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(all_docs)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            st.session_state.vectorstore = vectorstore
            st.session_state.messages = []

        st.sidebar.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} document(s)")

# ----------------------------------------------------------------------
# CHAT INTERFACE
# ----------------------------------------------------------------------
if st.session_state.vectorstore is None:
    st.info("👈 Upload PDF(s), enter your free Groq API key, and click **Process Documents** to begin.")
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📄 View sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"<div class='source-box'><b>{src['source']}</b> "
                            f"(page {src['page']})<br>{src['text']}</div>",
                            unsafe_allow_html=True,
                        )

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        if not groq_api_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    llm = ChatGroq(api_key=groq_api_key, model=model_name, temperature=0.2)
                    retriever = st.session_state.vectorstore.as_retriever(
                        search_kwargs={"k": top_k}
                    )
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=retriever,
                        return_source_documents=True,
                    )
                    result = qa_chain.invoke({"query": query})
                    answer = result["result"]
                    source_docs = result["source_documents"]

                st.markdown(answer)

                sources = []
                if source_docs:
                    with st.expander("📄 View sources"):
                        for doc in source_docs:
                            src_name = doc.metadata.get("source", "document")
                            page = doc.metadata.get("page", "N/A")
                            text_snippet = doc.page_content[:300].replace("\n", " ") + "..."
                            st.markdown(
                                f"<div class='source-box'><b>{src_name}</b> "
                                f"(page {page})<br>{text_snippet}</div>",
                                unsafe_allow_html=True,
                            )
                            sources.append({"source": src_name, "page": page, "text": text_snippet})

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )