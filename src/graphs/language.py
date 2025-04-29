"""The flow to identify a user's language and performs translations."""

from typing import TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableLambda

from ..resources import CHAT  # noqa: TID252


class LanguageState(TypedDict):
    """The state used in the language sub graph."""

    original: str
    """Original message as it was sent by the user."""
    language: str
    """The language that has been identified as being that of the user."""
    translated: str
    """A translation of the user's text to English."""


# The template used to determine the user's langage
_LANGUAGE = ChatPromptTemplate.from_messages([
    ("system", "What language does the user speak ? Anwer using the language name only."),
    ("user",   "{input}"),
])

def find_language(state: LanguageState) -> dict[str, str]:
    """Return the name of the language the user used to express its request."""
    language = ( _LANGUAGE | CHAT | StrOutputParser() ) .invoke({"input": state["original"]})
    return {"language": language}


LANGUAGE_GRAPH = RunnableLambda(find_language)
"""The actual language graph that you can use."""
