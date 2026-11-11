# RAG Document Search

A Retrieval-Augmented Generation (RAG) application that lets you search and query documents using natural language, powered by an LLM-backed retrieval pipeline with a Streamlit interface.

## 🎯 What it does
Index web pages and PDFs, then ask questions about them in plain English — the app retrieves the most relevant chunks and generates a grounded answer instead of relying on the LLM's raw knowledge.

- **Chat with follow-ups:** a question like "explain that more" is understood from the conversation.
- **Your own documents:** upload PDF, text or Markdown files in the sidebar.
- **Checkable answers:** each answer cites numbered passages, listed with the page title or PDF page number.
- **Streaming:** the answer appears as the model writes it.

## 🛠️ Tech Stack
- **Language:** Python (managed with `uv`)
- **UI:** Streamlit
- **RAG Framework:** LangChain / LangGraph
- **Search:** embeddings (`all-MiniLM-L6-v2`) in a cluster tree or FAISS, BM25 keyword search, and a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`), all running locally
- **LLM Provider:** Ollama (default, free and local), Groq, Hugging Face, or OpenAI

## 📁 Project Structure
```
RAGDOCUMENTSEARCH/
├── src/                # core RAG pipeline (ingestion, retrieval, generation)
├── data/                # source documents
├── evaluation/          # retrieval test questions, script and results
├── tests/               # unit tests (pytest)
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

## 🔎 Retrieval
A question goes through up to four steps. Steps 2 to 4 can be switched off in `.env`.

1. **Embedding search** finds the 20 chunks closest in meaning. By default the chunks are clustered into a tree and a query follows only the most similar branches. Set `RETRIEVER_TYPE=flat` to compare with every chunk using FAISS instead.
2. **Keyword search (BM25)** finds the 20 chunks that contain the question's exact words, which catches names, numbers and acronyms that embeddings miss. Off with `HYBRID_SEARCH=false`.
3. **Fusion** merges the two lists with reciprocal rank fusion.
4. **Reranking** has a cross-encoder read the question together with each candidate and keep the best 4. Off with `RERANK=false`.

Documents are split into chunks of 1,000 characters with 150 overlap (`CHUNK_SIZE`, `CHUNK_OVERLAP`).

**Saved index:** chunks and embeddings are stored in `index_cache/` after the first run, so later starts skip the download and embedding. The index is rebuilt automatically when the source list, a local file, or the chunk settings change. A changed web page is not detected, so use **Rebuild index** in the app sidebar or delete the `index_cache/` folder.

## 🧭 How a question is answered
The workflow is a LangGraph graph with a branch and a loop:

```
contextualize -> retriever -> grader --passages fit--> responder
                    ^            |
                    +- rewriter <+  passages don't fit
```

- **contextualize** rewrites a follow-up question so it makes sense on its own.
- **retriever** runs the search described above.
- **grader** asks the model whether the passages can answer the question.
- **rewriter** rewords the question for a second search when they can't. It runs once at most.
- **responder** is a ReAct agent that answers from the passages and can search the documents again or Wikipedia.

The grader costs one extra model call per question. Set `CHECK_RELEVANCE=false` to skip it.

## 📊 Evaluation
[evaluation/questions.json](evaluation/questions.json) holds 20 questions about the built-in documents, each with a phrase copied from the source text. A search counts as a hit when one of the 4 returned chunks contains that phrase.

| Chunks (size / overlap) | Index | Embeddings only | + keywords | + reranker | + keywords + reranker |
|---|---|---|---|---|---|
| 500 / 50 | flat | 65% | 70% | 75% | 75% |
| 500 / 50 | tree | 45% | 60% | 65% | 80% |
| 1,000 / 150 | flat | 75% | 80% | 85% | 90% |
| 1,000 / 150 | tree | 75% | 85% | 85% | **90%** (default) |

- The tree compares a query with 19% to 33% of the chunk embeddings. On its own it finds fewer answers than flat search with small chunks (45% against 65%), and keyword search plus reranking close that gap.
- Reranking takes the search from about 10 ms to about 0.9 s on a laptop CPU.
- Four 1,000-character chunks hold twice the text of four 500-character chunks, which is part of why larger chunks score higher.
- With 20 questions, one question is 5 points, so small differences are not meaningful.

The full table, with mean reciprocal rank and timings, is in [evaluation/results.md](evaluation/results.md). To run it again:
```bash
uv run python evaluation/evaluate.py
```

## ✅ Tests
```bash
uv run pytest
```
The tests cover BM25, both indexes, fusion, the saved index, document loading, and the graph's branch and loop. They use stand-ins for the models, so they need no API key.

## ☁️ Deployment
Ollama only runs on the machine where it is installed, so a hosted app needs a cloud provider such as Groq. `.env` isn't pushed to GitHub, so add the settings on the host as well.

**Streamlit Community Cloud:** add these under **Settings → Secrets**, then reboot the app:
```toml
LLM_PROVIDER = "groq"
GROQ_API_KEY = "your-key-here"
```

Uploaded files are stored in `uploads/` on the server and are searched by everyone who uses that copy of the app. On Streamlit Community Cloud they are lost when the app restarts.

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
1. Put PDFs in the `data/` folder and URLs in `data/url.txt` (one per line), or upload files in the app's sidebar
2. Launch the Streamlit app
3. Ask a question — the app retrieves relevant document chunks and returns a grounded answer with sources
4. Ask a follow-up, such as "why that many?"

## 📌 Future Improvements
- Add support for more file formats (DOCX, etc.)
- Keep each visitor's uploads separate on a hosted app
- Evaluate answer quality as well as retrieval
