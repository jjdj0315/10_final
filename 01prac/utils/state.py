from typing import TypedDict, List, Annotated
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class State(TypedDict):
    query: str
    documents: List[Document]
    answer: str
    messages: Annotated[List, add_messages]
    mode: str
