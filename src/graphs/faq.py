"""Necessary flows to search information in the FAQ."""
from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.string import StrOutputParser
from langgraph.graph import END, StateGraph

from ..resources import CHAT, FAQ  # noqa: TID252


class FaqState(TypedDict):
    """Tracks the state of every steps in the FAQ process."""

    english: str
    """Translation of the user's question in English."""
    questions: list[str]
    """The list of normalized questions in the user's query."""
    qid: int
    """The question that is currently being answered to."""
    sources: list[str]
    """The list of sources used to answer the user's queston."""
    partial: Annotated[list[str], add]
    """The set of partial attempts at answering the user's question."""
    response: str
    """The response that was given by the FAQ expert."""


class FaqInput(TypedDict):
    """Input of the FAQ graph."""

    english: str
    """Translation of the user's qustion in English."""


class FaqOutput(TypedDict):
    """Output of the FAQ graph."""

    response: str
    """The response that was given by the FAQ expert."""


# The template used to normalize the user's question.
_NORMALIZE = ChatPromptTemplate.from_messages([
    ("system", """
    You are an automated mail support assistant who extracts the key requests of an user
    and frames them as a list of questions such as the ones one could find in an FAQ.

    For instance, if an user said:
    '''
    I am a retired historian who always loved to go wandering in the reading rooms.
    I want to conduct some genealogical research to find my great grand parents: Gert en Lisa who died in 1912.
    I dont know where they were born, but I assume it must be somewhere near Zaventem. However I cannot proceed
    because the system tells me to connect but I dont know my password to your platform. Can you help me please ?

    Peter VAN CUREGHEM
    '''

    Then you should answer the following JSON list:
    [
    "How do I log in ?",
    "I forgot my password. What can I do ?",
    "How do I get started with genealogical research ?"
    ]
    '''
    """),
    ("user",   "{input}"),
])

_RESPOND = ChatPromptTemplate.from_messages([
    ("system", """
    You are a support assistant helping the users of the Belgian State Archives.
    Answer to the user's question using only the FAQ information below.
    Be as precise and accurate as can be. In case you don't know the answer, just say so.
    Be professional, and use a neutral tone.

    # FAQ
    {faq}
    """),
    ("user", "{question}"),
])

def normalize(state: FaqState) -> dict[str, str|int]:
    """Reoformulate the user input as a list of questions."""
    questions = ( _NORMALIZE | CHAT | JsonOutputParser() ) .invoke({"input": state["english"]})
    return {"questions": questions, "qid": 0}

def search(state: FaqState) -> dict[str, list[str]]:
    """Search for the current question using the FAQ."""
    qid      = state["qid"]
    question = state["questions"][qid]
    prompt   = f"""
    Instruct: Retrieve the FAQ entries that are most similar to the user's query.
    Query: {question}
    """
    sources = FAQ.as_retriever(search_method="mmr").invoke(prompt)
    sources = [ e.metadata["fulltext"] for e in sources ]
    return {"sources": sources}

def respond(state: FaqState) -> dict[str, list[str] | int]:
    """Respond to the current question using the FAQ."""
    qid      = state["qid"]
    question = state["questions"][qid]
    sources  = state["sources"]
    faq      = "\n\n".join(sources)
    prompt   = _RESPOND.invoke({
        "faq":      faq,
        "question": question,
    })
    response = (CHAT | StrOutputParser()).invoke(prompt)
    return {"partial": [response], "qid": qid+1}


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


def synthesize(state: FaqState) -> dict[str, str]:
    """Write a synthesis of the proposed items."""
    original = state["english"]
    partials = "\n\n".join(state["partial"])
    prompt   = _SYNTHEZIZE.invoke({"original": original, "partials": partials})
    response = (CHAT | StrOutputParser()).invoke(prompt)
    return {"response": response}


def finish_or_continue(state: FaqState) -> Literal["search", "synthesize"] :
    """Decide whether a next question should be processed."""
    qid = state["qid"]
    qs  = state["questions"]
    return "search" if qid < len(qs) else "synthesize"

_builder = StateGraph(FaqState, input=FaqInput, output=FaqOutput)
_builder.add_node("normalize",      normalize)
_builder.add_node("search",         search)
_builder.add_node("respond",        respond)
_builder.add_node("synthesize",     synthesize)

_builder.set_entry_point("normalize")
_builder.add_edge("normalize",     "search")
_builder.add_edge("search",        "respond")
_builder.add_conditional_edges("respond", finish_or_continue)
_builder.add_edge("synthesize", END)

FAQ_GRAPH = _builder.compile()
"""The actual faq graph that you can use"""
