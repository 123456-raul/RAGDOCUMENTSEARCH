"""LangGraph nodes for the RAG workflow.

contextualize -> retrieve -> grade -> answer, with a loop back to retrieve
through rewrite when the grader finds the passages don't fit the question.
The answer is written by a ReAct agent that can also search Wikipedia.
"""

import re
import time
from typing import Any, Dict, List, Optional

from src.config.config import Config
from src.document_ingestion.document_processor import source_label
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain.agents import create_agent

import requests

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks automated clients to identify themselves
WIKIPEDIA_USER_AGENT = "DocRAG/0.1 (local RAG app)"
WIKIPEDIA_RESULTS = 3
WIKIPEDIA_SUMMARY_CHARS = 1500

# LangGraph leaves model calls with this tag out of the token stream, so only
# the final answer is streamed to the user, not the helper calls
NOSTREAM_TAG = "nostream"

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


def message_text(message: Any) -> str:
    """The text of a model reply, without any <think> block"""
    content = getattr(message, "content", message)
    if isinstance(content, list):  # some providers return a list of parts
        content = "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
        )
    return _THINKING.sub("", str(content or "")).strip()


def format_passages(docs: List[Document]) -> str:
    """Number the passages and say where each one comes from"""
    return "\n\n".join(
        f"[{i}] ({source_label(doc)})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def wikipedia_search(query: str) -> str:
    """
    Return the opening summary of the best-matching Wikipedia pages.

    One request fetches the search results together with their summaries.
    Wikipedia answers a burst of requests with HTTP 429, which is what the
    `wikipedia` library ran into by sending several requests per page.
    """
    try:
        response = requests.get(
            WIKIPEDIA_API,
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrlimit": WIKIPEDIA_RESULTS,
                "prop": "extracts",
                "exintro": 1,
                "explaintext": 1,
                "exlimit": WIKIPEDIA_RESULTS,
                "redirects": 1,
            },
            headers={"User-Agent": WIKIPEDIA_USER_AGENT},
            timeout=10,
        )
        if response.status_code == 429:
            wait = response.headers.get("Retry-After", "a few")
            return f"Wikipedia is rate-limiting requests. Try again in {wait} seconds."
        response.raise_for_status()
        pages = response.json().get("query", {}).get("pages", {})
    except Exception as e:
        return f"Wikipedia search failed: {e}"

    # "index" is the page's position in the search results
    ranked = sorted(pages.values(), key=lambda page: page.get("index", 0))
    summaries = [
        f"Page: {page['title']}\nSummary: {page['extract'][:WIKIPEDIA_SUMMARY_CHARS]}"
        for page in ranked
        if page.get("extract")
    ]
    return "\n\n".join(summaries) or "No Wikipedia page found."


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent

    def _ask(self, prompt: str) -> str:
        """One short helper call to the model, kept out of the answer stream"""
        reply = self.llm.invoke(prompt, config={"tags": [NOSTREAM_TAG]})
        return message_text(reply)

    def _history(self, state: RAGState) -> List[Dict[str, str]]:
        """The most recent turns of the conversation"""
        return state.chat_history[-2 * Config.HISTORY_TURNS:]

    # ---- Nodes ----

    def contextualize(self, state: RAGState) -> dict:
        """
        Make a follow-up question self-contained.

        "Explain that more" can't be searched for. With the conversation in
        hand, the model rewrites it into a question that can.
        """
        history = self._history(state)
        if not history:
            return {"search_query": state.question}

        transcript = "\n".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content'][:500]}"
            for turn in history
        )
        rewritten = self._ask(
            "Rewrite the user's latest question so that it can be understood "
            "without the conversation. Replace words like 'it' or 'that' with "
            "what they refer to. If the question is already clear on its own, "
            "repeat it unchanged. Reply with the question only.\n\n"
            f"Conversation:\n{transcript}\n\n"
            f"Latest question: {state.question}"
        )
        return {"search_query": rewritten or state.question}

    def retrieve_docs(self, state: RAGState) -> dict:
        """Search the documents for the current search query"""
        start = time.perf_counter()
        docs = self.retriever.invoke(state.search_query or state.question)
        return {
            "retrieved_docs": docs,
            "retrieval_time": state.retrieval_time + time.perf_counter() - start,
        }

    def grade_docs(self, state: RAGState) -> dict:
        """Ask the model whether the retrieved passages can answer the question"""
        if not state.retrieved_docs:
            return {"docs_relevant": False}
        if not Config.CHECK_RELEVANCE:
            return {"docs_relevant": True}

        verdict = self._ask(
            "You are checking search results. Do the passages below contain "
            "information that helps answer the question? Reply with one word: "
            "yes or no.\n\n"
            f"Passages:\n{format_passages(state.retrieved_docs)}\n\n"
            f"Question: {state.search_query or state.question}"
        )
        # An unclear reply counts as relevant, so a confused grader costs nothing
        return {"docs_relevant": not verdict.lower().startswith("no")}

    def route_after_grading(self, state: RAGState) -> str:
        """Answer now, or rewrite the question and search again?"""
        if state.docs_relevant or state.rewrites >= Config.MAX_REWRITES:
            return "answer"
        return "rewrite"

    def rewrite_query(self, state: RAGState) -> dict:
        """Reword the question for another search"""
        query = state.search_query or state.question
        rewritten = self._ask(
            "A document search for the question below found nothing useful. "
            "Write a better search query for it: use other words for the same "
            "idea, spell out abbreviations, and drop filler words. Reply with "
            "the query only.\n\n"
            f"Question: {query}"
        )
        return {"search_query": rewritten or query, "rewrites": state.rewrites + 1}

    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            return format_passages(docs[:8])

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn
        )

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wikipedia_search,
        )

        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "The user's message includes numbered passages already retrieved from the indexed documents. "
            "Answer from those passages when they are enough, without calling a tool, "
            "and cite the ones you used by number, like [1]. "
            "Call 'retriever' only to search the documents with a different query, "
            "and 'wikipedia' only for general knowledge the passages don't cover. "
            "Say so when an answer comes from Wikipedia or from your own knowledge "
            "instead of the passages. "
            "Return only the final useful answer."
        )
        self._agent = create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> dict:
        """
        Generate answer using ReAct agent with retriever + wikipedia.

        The passages found by the retriever node are passed in the message,
        so the agent can usually answer in a single model call. Earlier turns
        of the conversation come first, so follow-up questions make sense.
        """
        if self._agent is None:
            self._build_agent()

        messages: List[BaseMessage] = [
            HumanMessage(content=turn["content"])
            if turn["role"] == "user"
            else AIMessage(content=turn["content"])
            for turn in self._history(state)
        ]

        message = state.question
        if state.retrieved_docs:
            intro = (
                "Passages from the indexed documents:"
                if state.docs_relevant
                else "Passages from the indexed documents (a check found they may not answer the question):"
            )
            message = (
                f"{intro}\n{format_passages(state.retrieved_docs)}\n\n"
                f"Question: {state.question}"
            )
        messages.append(HumanMessage(content=message))

        result = self._agent.invoke({"messages": messages})

        replies = result.get("messages", [])
        answer: Optional[str] = message_text(replies[-1]) if replies else None

        return {"answer": answer or "Could not generate answer."}
