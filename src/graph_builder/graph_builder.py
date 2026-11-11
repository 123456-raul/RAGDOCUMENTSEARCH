"""Graph builder for LangGraph workflow"""

from typing import Dict, Iterator, List, Optional, Tuple

from langchain_core.messages import AIMessageChunk
from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize graph builder

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        """
        Build the RAG workflow graph

            contextualize -> retriever -> grader --answer--> responder -> END
                                ^            |
                                +- rewriter <+ (passages don't fit the question)

        Returns:
            Compiled graph instance
        """
        # Create state graph
        builder = StateGraph(RAGState)

        # Add nodes
        builder.add_node("contextualize", self.nodes.contextualize)
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("grader", self.nodes.grade_docs)
        builder.add_node("rewriter", self.nodes.rewrite_query)
        builder.add_node("responder", self.nodes.generate_answer)

        # Set entry point
        builder.set_entry_point("contextualize")

        # Add edges
        builder.add_edge("contextualize", "retriever")
        builder.add_edge("retriever", "grader")
        # The branch: answer, or rewrite the question and search again (the loop)
        builder.add_conditional_edges(
            "grader",
            self.nodes.route_after_grading,
            {"answer": "responder", "rewrite": "rewriter"},
        )
        builder.add_edge("rewriter", "retriever")
        builder.add_edge("responder", END)

        # Compile graph
        self.graph = builder.compile()
        return self.graph

    def _initial_state(
        self, question: str, chat_history: Optional[List[Dict[str, str]]]
    ) -> RAGState:
        return RAGState(question=question, chat_history=chat_history or [])

    def run(
        self, question: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> dict:
        """
        Run the RAG workflow

        Args:
            question: User question
            chat_history: Earlier turns as {"role", "content"} dicts, oldest first

        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()

        return self.graph.invoke(self._initial_state(question, chat_history))

    def stream(
        self, question: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Iterator[Tuple[str, object]]:
        """
        Run the workflow and report progress as it happens

        Yields:
            ("step", node name) when a node finishes,
            ("token", text) for each piece of the answer as the model writes it,
            ("result", final state) once at the end
        """
        if self.graph is None:
            self.build()

        state: dict = {}
        # subgraphs=True because the answer is written inside the responder's
        # agent, which is a graph of its own; `namespace` is empty for this graph
        for namespace, mode, payload in self.graph.stream(
            self._initial_state(question, chat_history),
            stream_mode=["messages", "updates", "values"],
            subgraphs=True,
        ):
            if mode == "messages":
                chunk, _ = payload
                # Only the answer's own text: no tool calls or tool results
                if (
                    isinstance(chunk, AIMessageChunk)
                    and isinstance(chunk.content, str)
                    and chunk.content
                    and not chunk.tool_call_chunks
                ):
                    yield "token", chunk.content
            elif namespace:
                continue  # steps inside the agent
            elif mode == "updates":
                for node_name in payload:
                    yield "step", node_name
            else:
                state = payload

        yield "result", state
