"""Various flows that are used to compose the agentic digita mailbox manager."""

from operator import add
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from .faq import FAQ_GRAPH
from .language import LANGUAGE_GRAPH
from .synthesize import SYNTHESIZE_GRAPH
from .translate import TRANSLATE_GRAPH


class State(TypedDict):
    """The state to manage the main overarching process."""

    original: str
    """The original message that has been sent by the user."""
    language: str
    """The language in which the user has written his question."""
    to_translate: str
    """The text that must be translated to another language."""
    to_language: str
    """The destination language for translation."""
    translated: str
    """The translation of the user's message in english."""
    partial: Annotated[list[str], add]
    """The list of partial responses that have been given to the user along different flows."""
    sythesis: str
    """The synthesis of all partial responses;"""
    response: str
    """The final answer that is given to the user."""


class Input(TypedDict):
    """Input schema for the main overarching graph."""

    original: str
    """The original email / message that was sent by the user."""


class Output(TypedDict):
    """Output schema for the main overarching graph."""

    response: str
    """The response brought to the user by the system."""

def to_english(state: State) -> dict[str, str]:
    """Configure the translation step to produce english."""
    return {
        "to_language": "English",
        "to_translate": state["original"],
    }

def to_user_language(state: State) -> dict[str, str]:
    """Configure the translation step to produce the user's language."""
    return {
        "to_language": state["language"],
        "to_translate": state["sythesis"],
    }

def reply_to_user(state: State) -> dict[str, str]:
    """Give the final answer to the user (and possibly send an actual email out)."""
    return {"response": state["translated"]}

_builder = StateGraph(State, input=Input, output=Output)
_builder.add_node("detect_language",   LANGUAGE_GRAPH)
_builder.add_node("to_english",        to_english)
_builder.add_node("translate_eng",     TRANSLATE_GRAPH)
_builder.add_node("to_user_language",  to_user_language)
_builder.add_node("translate_usr",     TRANSLATE_GRAPH)
_builder.add_node("faq",               FAQ_GRAPH)
_builder.add_node("synthesize",        SYNTHESIZE_GRAPH)
_builder.add_node("reply_to_user",     reply_to_user)

_builder.add_edge(START,                      "detect_language")
_builder.add_edge(START,                      "to_english")
_builder.add_edge("to_english",               "translate_eng")
_builder.add_edge("translate_eng",            "faq")
_builder.add_edge(["faq", "detect_language"], "synthesize")
_builder.add_edge("synthesize",               "to_user_language")
_builder.add_edge("to_user_language",         "translate_usr")
_builder.add_edge("translate_usr",         "reply_to_user")
_builder.add_edge("reply_to_user",            END)

ARKEY_GRAPH = _builder.compile()
