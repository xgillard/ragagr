"""The flow to identify a user's language and performs translations."""

from typing import TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langgraph.graph import END, START, StateGraph

from ..resources import CHAT  # noqa: TID252


class TranslateState(TypedDict):
    """The state used in the language sub graph."""

    to_translate: str
    """Original message as it was sent by the user."""
    to_language: str
    """The language to which the original text must be translated."""
    translated: str
    """A translation of the user's text to English."""


# The template used to normalize the user's question.
_TRANSLATE = ChatPromptTemplate.from_messages([
    ("system", """
    You are an automated translation assistant who translates email to {to_language}.
    Please do not change the content of the email and just write the translation.
    """),
    ("user", "{to_translate}"),
])

def translate_to(state: TranslateState) -> dict[str, str]:
    """Translate the text 'to_translate' to the target language 'to_language."""
    prompt = _TRANSLATE.invoke({
        "to_translate": state["to_translate"],
        "to_language":  state["to_language"],
    })
    translated = (CHAT | StrOutputParser() ) .invoke(prompt)
    return {"translated": translated}


TRANSLATE_GRAPH = translate_to
"""The actual translation graph that you can use"""
