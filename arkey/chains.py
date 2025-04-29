"""The chains to implement the various steps of the arkey workflow."""
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from .prompts import (
    DETERMINE_LANGUAGE,
    RESPOND,
    ROUTE_TO_FLOW,
    SPLIT_REQUESTS,
    SYNTHESIZE,
    TRANSLATE,
)
from .resources import CHAT

find_language = (
    {"input": lambda x: x}
    | DETERMINE_LANGUAGE
    | CHAT
    | StrOutputParser()
)
"""Return the name of the language the user used to express its request."""

translate = (
    TRANSLATE # input {"input", "target"}
    | CHAT
    | StrOutputParser()
)
"""Return a translation of the given input."""

class QuestionsList(BaseModel):
    """The user's request reframed as a list of atomic questions."""

    questions: list[str]
    """The list of atomic questions"""

split_user_request = (
    {"input": lambda x: x}
    | SPLIT_REQUESTS
    | CHAT.with_structured_output(QuestionsList)
)
"""Return a QuestionsList of user requests."""

route_to_flow = (
    {"input": lambda x: x}
    | ROUTE_TO_FLOW
    | CHAT
    | StrOutputParser()
)
"""Return 'faq', 'genealogy', or 'records'."""

respond = (
    RESPOND # input {"facts", "question"}
    | CHAT
    | StrOutputParser()
)
"""Return a partial response to the user's request."""

synthesize = (
    SYNTHESIZE # input {"original", "partials"}
    | CHAT
    | StrOutputParser()
)
"""Return a final response to the user's request."""
