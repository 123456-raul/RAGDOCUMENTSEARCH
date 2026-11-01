# RAG Document Search

A Retrieval-Augmented Generation (RAG) application that lets you search and query documents using natural language, powered by an LLM-backed retrieval pipeline with a Streamlit interface.

## 🎯 What it does
Upload/index documents and ask questions about them in plain English — the app retrieves the most relevant chunks and generates a grounded answer instead of relying on the LLM's raw knowledge.

## 🛠️ Tech Stack
- **Language:** Python (managed with `uv`)
- **UI:** Streamlit
- **RAG Framework:** LangChain / LangGraph
- **Vector Store:** FAISS
- **LLM Provider:** Ollama (default, free and local), Groq, Hugging Face, or OpenAI

## 📁 Project Structure
```
RAGDOCUMENTSEARCH/
├── src/                # core RAG pipeline (ingestion, retrieval, generation)
├── data/                # source documents
├── main.py             # pipeline entry point
├── streamlit_app.py     # Streamlit UI
├── pyproject.toml       # project dependencies (uv)
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/download) for the default local model

### Installation
```bash
git clone https://github.com/123456-raul/RAGDOCUMENTSEARCH.git
cd RAGDOCUMENTSEARCH
uv sync
ollama pull qwen3:4b
```

### Environment Variables
Ollama needs no keys. To use another provider, create a `.env` file with the provider and its key:
```
LLM_PROVIDER=groq   # ollama | groq | huggingface | openai
GROQ_API_KEY=your-key-here
HUGGINGFACEHUB_API_TOKEN=your-token-here
OPENAI_API_KEY=your-key-here
```

### Run the App
```bash
uv run streamlit run streamlit_app.py
```

Or run the pipeline directly:
```bash
uv run python main.py
```

## 💡 Example Usage
1. Put PDFs in the `data/` folder and URLs in `data/url.txt` (one per line)
2. Launch the Streamlit app
3. Ask a question — the app retrieves relevant document chunks and returns a grounded answer with sources

## 📌 Future Improvements
- Add support for multiple file formats (DOCX, etc.)
- Add source citation display in the UI
- Add evaluation metrics for retrieval quality
