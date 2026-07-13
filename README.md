# 🧠 DocMind — RAG-Powered Document Q&A

Ask anything about your PDF. DocMind is an agentic Retrieval-Augmented Generation (RAG) pipeline that lets you upload any PDF and get accurate, context-grounded answers in real time — no hallucinated fluff, just answers pulled directly from your document.

🔗 **Live Demo:** [vvek2611-docmind-app-zjflde.streamlit.app](https://vvek2611-docmind-app-zjflde.streamlit.app)

---

## ✨ Features

- 📄 **Upload any PDF** — automatically chunked and embedded for retrieval
- 💬 **Conversational Q&A** — ask follow-up questions with full chat history
- ⚡ **Fast inference** — powered by Groq's LPU (Llama 3.3-70B) for near-instant responses
- 🔍 **Grounded answers** — retrieval-augmented, not just LLM guesswork
- ✅ **Evaluated pipeline** — includes a golden QA evaluation harness to measure retrieval/answer quality

---

## 🏗️ Architecture



PDF Upload → Chunking → Embedding → FAISS Vector Store
↓
User Query → Retriever → Top-K Chunks → Groq LLM (Llama 3.3-70B) → Answer

**Pipeline stages:**
1. **Ingest** — PDF is parsed and split into overlapping chunks
2. **Embed** — Chunks are embedded and indexed in a FAISS vector store
3. **Retrieve** — User query is embedded and matched against the index
4. **Generate** — Retrieved context + query are passed to the LLM for a grounded answer

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangChain |
| Vector Store | FAISS |
| LLM | Groq (Llama 3.3-70B) |
| Backend | FastAPI *(where applicable)* |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |
| Evaluation | Custom golden QA harness (`eval_harness.py`) |

---

## 🚀 Run Locally

```bash
git clone https://github.com/Vvek2611/docmind.git
cd docmind
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the root:

