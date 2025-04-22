"""The steps of a synthezize flow which synthesizes the output of several different subflows."""

from operator import add
from typing import Annotated, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langgraph.graph import END, START, StateGraph

from ..resources import CHAT  # noqa: TID252


class SynthezizeState(TypedDict):
    """Tracks the state of every steps in the FAQ process."""

    translated: str
    """Translation of the user's question in English."""
    partial: Annotated[list[str], add]
    """The set of partial attempts at answering the user's question."""
    sythesis: str
    """The final message to send to the user in order to answer its question."""

_SYNTHEZIZE = ChatPromptTemplate.from_template("""
    You are a support email assistant who synthezizes the inputs from colleagues and use these inputs to
    write an email answering the user's request. Always be polite, professional and use a neutral tone.
    Be accurate, do not invent facts that are not backed by the inputs from your colleagues.
    When you cannot satisfy a user's request, just say so.
    At the end of your email, sign as 'Arkey'.

    # User's request
    {original}

    # Input from your colleagues
    {partials}
    """)


def synthesize(state: SynthezizeState) -> dict[str, str]:
    """Write a synthesis of the proposed items."""
    original = state["translated"]
    partials = "\n\n".join(state["partial"])
    prompt   = _SYNTHEZIZE.invoke({"original": original, "partials": partials})
    response = (CHAT | StrOutputParser()).invoke(prompt)
    return {"synthesis": response}


SYNTHESIZE_GRAPH = synthesize
"""The actual synthezize graph that you can use"""
