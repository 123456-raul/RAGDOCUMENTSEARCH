# RAG Document Search

A Retrieval-Augmented Generation (RAG) application that lets you search and query documents using natural language, powered by an LLM-backed retrieval pipeline with a Streamlit interface.

## 🎯 What it does
Upload/index documents and ask questions about them in plain English — the app retrieves the most relevant chunks and generates a grounded answer instead of relying on the LLM's raw knowledge.

## 🛠️ Tech Stack
- **Language:** Python (managed with `uv`)
- **UI:** Streamlit
- **RAG Framework:** <!-- e.g. LangChain / LlamaIndex — fill in -->
- **Vector Store:** <!-- e.g. FAISS / Chroma / Pinecone — fill in -->
- **LLM Provider:** <!-- e.g. OpenAI / Groq / HuggingFace — fill in -->

## 📁 Project Structure
\`\`\`
RAGDOCUMENTSEARCH/
├── src/                # core RAG pipeline (ingestion, retrieval, generation)
├── data/                # source documents
├── main.py             # pipeline entry point
├── streamlit_app.py     # Streamlit UI
├── pyproject.toml       # project dependencies (uv)
└── requirements.txt
\`\`\`

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- [uv](https://docs.astral.sh/uv/) package manager

### Installation
\`\`\`bash
git clone https://github.com/123456-raul/RAGDOCUMENTSEARCH.git
cd RAGDOCUMENTSEARCH
uv sync
\`\`\`

### Environment Variables
Create a `.env` file with your API keys:
\`\`\`
<!-- e.g. OPENAI_API_KEY=your-key-here -->
\`\`\`

### Run the App
\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

Or run the pipeline directly:
\`\`\`bash
python main.py
\`\`\`

## 💡 Example Usage
1. Place your documents in the `data/` folder
2. Launch the Streamlit app
3. Ask a question — the app retrieves relevant document chunks and returns a grounded answer with sources

## 📌 Future Improvements
- Add support for multiple file formats (PDF, DOCX, etc.)
- Add source citation display in the UI
- Add evaluation metrics for retrieval quality
