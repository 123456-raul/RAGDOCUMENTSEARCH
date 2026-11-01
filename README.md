# RAG Document Search

A Retrieval-Augmented Generation (RAG) application that lets you search and query documents using natural language, powered by an LLM-backed retrieval pipeline with a Streamlit interface.

## 🎯 What it does
Index web pages and PDFs, then ask questions about them in plain English — the app retrieves the most relevant chunks and generates a grounded answer instead of relying on the LLM's raw knowledge.

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
- Python 3.10 to 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.com/download) to run the default model locally (not needed if you use a cloud provider)

### Installation
```bash
git clone https://github.com/123456-raul/RAGDOCUMENTSEARCH.git
cd RAGDOCUMENTSEARCH
uv sync
ollama pull qwen3:4b   # skip if you use a cloud provider
```

### Choose an LLM Provider
Set `LLM_PROVIDER` in a `.env` file in the project root. If it isn't set, the app uses Ollama.

| `LLM_PROVIDER` | Cost | Key needed | Default model | Change the model with |
|---|---|---|---|---|
| `ollama` | Free, runs locally | None | `qwen3:4b` | `OLLAMA_MODEL` |
| `groq` | Free tier with rate limits | `GROQ_API_KEY` | `qwen/qwen3.8-27b` | `GROQ_MODEL` |
| `huggingface` | Paid credits may be needed | `HUGGINGFACEHUB_API_TOKEN` | `Qwen/Qwen2.5-1.5B-Instruct` | `HF_LLM_REPO_ID` |
| `openai` | Paid | `OPENAI_API_KEY` | `gpt-4o-mini` | `OPENAI_MODEL` |

For example, to use Groq with a [free API key](https://console.groq.com/keys):
```
LLM_PROVIDER=groq
GROQ_API_KEY=your-key-here
```

Never commit `.env`. It is listed in `.gitignore`, and any key that has been committed should be replaced.

### Run the App
```bash
uv run streamlit run streamlit_app.py
```

Or run the pipeline directly:
```bash
uv run python main.py
```

Restart the app after changing `.env`, because settings are only read at startup.

## ☁️ Deployment
Ollama only runs on the machine where it is installed, so a hosted app needs a cloud provider such as Groq. `.env` isn't pushed to GitHub, so add the settings on the host as well.

**Streamlit Community Cloud:** add these under **Settings → Secrets**, then reboot the app:
```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-key-here"
```

**GitHub Codespaces:** create a `.env` file in the Codespace with the same settings, written as in the Groq example above, then restart the app.

## 🔧 Troubleshooting
| Error | Cause | Fix |
|---|---|---|
| `Could not get an answer from the ollama model: [Errno 111] Connection refused` | The app can't reach Ollama, usually because it is hosted and has no `LLM_PROVIDER` setting | Follow the Deployment steps above |
| `WinError 10061` or another Ollama connection error on your own machine | Ollama isn't running | Open the Ollama app or run `ollama serve` |
| `model '...' not found` from Ollama | The model hasn't been downloaded | Run `ollama pull <model>` |
| `model_not_found` from Groq | The model isn't available to your account | Set `GROQ_MODEL` to one listed in the [Groq console](https://console.groq.com/docs/models) |
| `model_not_supported` from Hugging Face | No enabled provider serves the model | Enable a provider in your [inference settings](https://huggingface.co/settings/inference-providers) or choose another model |

## 💡 Example Usage
1. Put PDFs in the `data/` folder and URLs in `data/url.txt` (one per line)
2. Launch the Streamlit app
3. Ask a question — the app retrieves relevant document chunks and returns a grounded answer with sources

## 📌 Future Improvements
- Add support for multiple file formats (DOCX, etc.)
- Add source citation display in the UI
- Add evaluation metrics for retrieval quality
