"""Necessary flows to search information in the getting started with genealogy."""
from __future__ import annotations

from operator import add
from typing import Annotated, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from ..resources import CHAT, GENEALOGY  # noqa: TID252


class GenealogyState(TypedDict):
    """Tracks the step in the genealogy search."""

    english: str
    """Translation of the user's question in English."""
    sources: list[str]
    """The list of sources used to answer the user's queston."""
    partial: Annotated[list[str], add]
    """The set of partial attempts at answering the user's question."""
    response: str
    """The synthesis of all partial responses"""

def search_genealogy(state: GenealogyState) -> dict[str, list[str]]:
    """Search in the getting started with genealogy information."""
    genea = GENEALOGY.as_retriever(search_type="mmr")
    prompt= f"""
    Instruct: Retrieve the passages most relevant to answer the user's email.
    User's Email: {state['english']}""".strip()
    results = [d.page_content for d in genea.get_relevant_documents(prompt) ]
    return {"sources": results}

__RESPOND_PROMPT = ChatPromptTemplate.from_template("""
    You are an expert who helps answering the emails from customers
    who may need getting started with their genealogical research. 
    Below, you will find some excepts from the official
    "getting started with genealogy" guide which you might use to answer.
    You will also find a copy of the user's email and a tentative
    response that was given by a colleague of yours who is not as
    much of a specialist as you are (he only knowss the FAQ of the 
    archive's website).

    If you think it is necessary, adapt the tentative reply to better
    answer the user's email. If you do not know the response to the
    user request, do not reply with made up information.

    # Excerpts from the official doc
    {sources}

    # Customer Email:
    {english}

    # Tentative reply based on the FAQ only
    {response}
""")
def respond(state: GenealogyState) -> dict[str, list[str] | int]:
    """Respond to the current question using the Genealogy getting started."""
    sources  = state["sources"]
    sources  = "\n\n".join(sources)
    prompt   = __RESPOND_PROMPT.invoke({
        "sources": sources,
        "english": state["english"],
        "response":  state["response"],
    })
    response = (CHAT | StrOutputParser()).invoke(prompt)
    return {"response": response}


_builder = StateGraph(GenealogyState, input=GenealogyState, output=GenealogyState)
_builder.add_node("search_genealogy", search_genealogy)
_builder.add_node("respond",          respond)
_builder.add_edge("search_genealogy", "respond")
_builder.add_edge("respond",          END)

_builder.set_entry_point ("search_genealogy")

GENEALOGY_GRAPH = _builder.compile()
"""The actual genealogy graph that you can use"""
