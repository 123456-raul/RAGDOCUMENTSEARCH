"""Streamlit chat UI for the Agentic RAG system"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import (
    DocumentProcessor,
    source_label,
    source_url,
)
from src.vectorstore.vectorstore import VectorStore
from src.vectorstore.index_cache import IndexCache
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# What to tell the user once each step of the graph has finished
STEP_MESSAGES = {
    "retriever": "Searched the documents",
    "grader": "Checked that the passages fit the question",
    "rewriter": "Passages didn't fit, so reworded the question to search again",
}


@st.cache_resource(show_spinner=False)
def initialize_rag():
    """Initialize the RAG system (cached; failures raise so they aren't cached)"""
    # Initialize components
    llm = Config.get_llm()
    doc_processor = DocumentProcessor(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP
    )
    vector_store = VectorStore()

    # Load the saved index, or download and embed the configured sources
    sources = Config.get_sources()
    num_chunks, from_cache = vector_store.load_or_build(sources, doc_processor)
    vector_store.warm_up()

    # Build graph
    graph_builder = GraphBuilder(
        retriever=vector_store.get_retriever(),
        llm=llm
    )
    graph_builder.build()

    return graph_builder, num_chunks, from_cache


def reset_index():
    """Forget the loaded system, so the next run indexes the sources again"""
    initialize_rag.clear()


def save_uploads(files) -> int:
    """Save uploaded files into the upload folder and return how many were saved"""
    Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for file in files:
        # Path(...).name drops any folder part, so a file can't land elsewhere
        target = Config.UPLOAD_DIR / Path(file.name).name
        target.write_bytes(file.getbuffer())
    return len(files)


def render_sidebar():
    """Document upload, the list of indexed sources, and the reset buttons"""
    with st.sidebar:
        st.markdown("### 📂 Your documents")
        with st.form("upload_form", clear_on_submit=True):
            files = st.file_uploader(
                "Add PDF, text or Markdown files",
                type=list(Config.UPLOAD_TYPES),
                accept_multiple_files=True,
            )
            if st.form_submit_button("➕ Add to index") and files:
                count = save_uploads(files)
                reset_index()
                st.toast(f"Added {count} file(s). Indexing…")

        for path in Config.get_uploads():
            name_col, button_col = st.columns([5, 1])
            name_col.caption(path.name)
            if button_col.button("✕", key=f"remove_{path.name}", help=f"Remove {path.name}"):
                path.unlink(missing_ok=True)
                reset_index()
                st.rerun()

        with st.expander("Built-in sources"):
            for source in Config.get_sources():
                if source.startswith("http"):
                    st.caption(source)
            for pdf in sorted(Config.DATA_DIR.glob("*.pdf")):
                st.caption(f"data/{pdf.name}")

        st.markdown("### ⚙️ Index")
        if st.button(
            "🔄 Rebuild index",
            help="Download the sources again and re-embed them",
        ):
            IndexCache(Config.INDEX_CACHE_DIR).clear()
            reset_index()

        if st.button("🧹 Clear chat"):
            st.session_state.messages = []

        search = "embeddings + keywords" if Config.HYBRID_SEARCH else "embeddings"
        if Config.RERANK:
            search += " + reranker"
        st.caption(
            f"Search: {Config.RETRIEVER_TYPE} index, {search}  \n"
            f"Model: {Config.LLM_PROVIDER}"
        )


def render_sources(sources):
    """List the passages an answer was based on, with where each came from"""
    if not sources:
        return
    with st.expander(f"📄 Sources ({len(sources)})"):
        for i, source in enumerate(sources, 1):
            label = source["label"]
            if source["url"]:
                label = f"[{label}]({source['url']})"
            st.markdown(f"**[{i}]** {label}")
            st.caption(source["text"])


def render_details(message):
    """Sources and timing shown under an assistant message"""
    render_sources(message.get("sources"))
    if "time" in message:
        note = (
            f"⏱️ {message['time']:.1f} s "
            f"(document search: {message['search_ms']:.0f} ms, the rest is the model)"
        )
        if message.get("search_query"):
            note += f"  \n🔎 Searched for: {message['search_query']}"
        st.caption(note)


def answer_question(rag_system, question):
    """Run the graph, showing progress and the answer as it is written"""
    # Earlier turns only; the new question is passed separately
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    start_time = time.time()
    status = st.status("Working…", expanded=False)
    placeholder = st.empty()
    streamed = ""
    result = {}

    try:
        for kind, payload in rag_system.stream(question, history):
            if kind == "step" and payload in STEP_MESSAGES:
                status.write(STEP_MESSAGES[payload])
                status.update(label=STEP_MESSAGES[payload] + "…")
            elif kind == "token":
                streamed += payload
                placeholder.markdown(streamed + "▌")
            elif kind == "result":
                result = payload
    except Exception as e:
        status.update(label="Failed", state="error")
        placeholder.empty()
        st.error(f"Could not get an answer from the {Config.LLM_PROVIDER} model: {str(e)}")
        return None

    elapsed_time = time.time() - start_time
    status.update(label=f"Done in {elapsed_time:.1f} s", state="complete")

    # The final answer replaces the streamed text, which can include words the
    # model wrote before deciding to call a tool
    answer = result.get("answer") or streamed or "Could not generate answer."
    placeholder.markdown(answer)

    search_query = result.get("search_query", "")
    message = {
        "role": "assistant",
        "content": answer,
        "sources": [
            {
                "label": source_label(doc),
                "url": source_url(doc),
                "text": doc.page_content[:400] + ("…" if len(doc.page_content) > 400 else ""),
            }
            for doc in result.get("retrieved_docs", [])
        ],
        "time": elapsed_time,
        "search_ms": result.get("retrieval_time", 0.0) * 1000,
        # Only worth showing when it differs from what the user typed
        "search_query": search_query if search_query != question else "",
    }
    render_details(message)
    return message


def main():
    """Main application"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents, or add your own in the sidebar")

    render_sidebar()

    # Initialize system (instant after the first run, until the index is reset)
    rag_system = None
    with st.spinner("Loading documents and models…"):
        try:
            rag_system, num_chunks, from_cache = initialize_rag()
            origin = "from the saved index" if from_cache else "downloaded and embedded"
            st.caption(f"✅ Ready: {num_chunks} document chunks, {origin}")
        except Exception as e:
            st.error(f"Failed to initialize: {str(e)}")

    # The conversation so far
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_details(message)

    question = st.chat_input(
        "Ask a question, then follow up on the answer",
        disabled=rag_system is None,
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            message = answer_question(rag_system, question)

        if message is None:
            # Drop the unanswered question, so it isn't sent as history later
            st.session_state.messages.pop()
        else:
            st.session_state.messages.append(message)

if __name__ == "__main__":
    main()
