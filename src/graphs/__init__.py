"""Various flows that are used to compose the agentic digita mailbox manager."""

from difflib import get_close_matches
from typing import Literal, TypedDict

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph

from ..resources import CHAT  # noqa: TID252
from .faq import FAQ_GRAPH
from .genealogy import GENEALOGY_GRAPH
from .language import LANGUAGE_GRAPH
from .translate import TRANSLATE_GRAPH


class State(TypedDict):
    """The state to manage the main overarching process."""

    original: str
    """The original message that has been sent by the user."""
    language: str
    """The language in which the user has written his question."""
    english: str
    """The translation of the user's message in english."""
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
    translation = TRANSLATE_GRAPH.invoke({
        "to_language":  "English",
        "to_translate": state["original"],
    })
    return {"english": translation["translated"]}

def to_user_language(state: State) -> dict[str, str]:
    """Configure the translation step to produce the user's language."""
    translation = TRANSLATE_GRAPH.invoke({
        "to_language": state["language"],
        "to_translate": state["response"],
    })
    return {"response": translation["translated"]}


__SORTING_PROMPT = ChatPromptTemplate.from_template("""
    You are an automated assistant to the Belgian State Archive who analyses
    customer emails and routes them towards the most knowledgeable colleague.
    Given a transcript of the user's email, and a tentative response from a colleague
    of yours who only knows about the FAQ, your role is to determine who would be the
    most appropriate to refine the tentative answer.

    Given the email below, reply one of the following literals:
    * "getting_started_genealogy" when the user needs help to kickstart its genealogical search.
    * "user_guide_genealogy" when the user needs practical guidance to search someone in the system.
    * "user_guide_records" when the user needs practical guidance on how to analyse records.

    # Customer Email:
    {english}
""")

__Expert = Literal["getting_started_genealogy", "user_guide_genealogy", "user_guide_records"]
def route_to_specialist(state: State) -> __Expert:
    """Route the message to a specialist agent."""
    prompt  = __SORTING_PROMPT.invoke(state)
    route   =  (CHAT | StrOutputParser()).invoke(prompt)
    route   = get_close_matches(route, ["getting_started_genealogy", "user_guide_genealogy", "user_guide_records"], n=1)
    return route[0]


__FIX_LINKS = ChatPromptTemplate.from_template("""
Your role is to fix urls from emails that have been written by assistants that did not
have an access to the internet. Whenever one of your colleague made references to
one of the following pages and inserted a placeholder for the actual URL, you can replace
that placeholder using the actual url:

* FAQ: https://agatha.arch.be/help/
* Genealogy, Getting Started: https://agatha.arch.be/help/beginning/
* Genealogy, User Guide: https://agatha.arch.be/help/States-Archives_user-guide-genealogical-sources.pdf
* Genealogy, Search Engine: https://agatha.arch.be/search/genealogie/ 
* Records, User Guide: https://agatha.arch.be/help/States-Archives_user-guide-analyses-records.pdf
* Records, Search Engine: https://agatha.arch.be/search/analyses/
* Inventories, Search Engine: https://agatha.arch.be/search/ead/
* Contact form: https://agatha.arch.be/contact/

# Response from your colleague
{response}
""")
def fix_links(state: State) -> dict[str, Literal["search_genealogy", "done"]]:
    """Route the message to a specialist agent."""
    prompt   = __FIX_LINKS.invoke(state)
    response =  (CHAT | StrOutputParser()).invoke(prompt)
    return {"response": response}


_builder = StateGraph(State, input=Input, output=Output)

_builder.add_node("detect_language",            LANGUAGE_GRAPH)
_builder.add_node("to_english",                 to_english)
_builder.add_node("faq",                        FAQ_GRAPH)

_builder.add_node("getting_started_genealogy",  GENEALOGY_GRAPH)
_builder.add_node("user_guide_genealogy",       lambda x: {"response": f"{x['response']}\n+++USR GUIDE G+++"})
_builder.add_node("user_guide_records",         lambda x: {"response": f"{x['response']}\n+++USR GUIDE R+++"})

_builder.add_node("fix_links",                  fix_links)
_builder.add_node("to_user_language",           to_user_language)

_builder.add_edge(START,                        "detect_language")
_builder.add_edge(START,                        "to_english")
_builder.add_edge("to_english",                 "faq")
_builder.add_conditional_edges("faq",           route_to_specialist)

_builder.add_edge("getting_started_genealogy",  "fix_links")
_builder.add_edge("user_guide_genealogy",       "fix_links")
_builder.add_edge("user_guide_records",         "fix_links")

_builder.add_edge(["detect_language", "fix_links"], "to_user_language")
_builder.add_edge("to_user_language",            END)

ARKEY_GRAPH = _builder.compile()
