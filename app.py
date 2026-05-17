import streamlit as st
import tempfile, os
from rag_pipeline import process_document

st.set_page_config(page_title="DocMind", page_icon="🧠", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0f0f1a; color: #e0e0e0; }
.source-box {
    background: #1a1a2e; border-left: 3px solid #1A56DB;
    padding: 10px; border-radius: 6px;
    font-size: 13px; color: #aaaacc; margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 DocMind")
st.caption("RAG-powered Document Q&A — Ask anything about your PDF")

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF\n2. Document is chunked & embedded\n3. Your question retrieves relevant chunks\n4. LLM answers using only those chunks")

if uploaded_file:
    with st.spinner("⚙️ Processing document..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        chain, num_chunks = process_document(tmp_path)
        os.unlink(tmp_path)

    st.success(f"✅ Document ready — {num_chunks} chunks indexed")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if query := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(query)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("👈 Upload a PDF from the sidebar to get started")