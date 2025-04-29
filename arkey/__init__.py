"""The main logic behind the Arkey mail bot."""
from __future__ import annotations

from operator import add
from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from .chains import (
    find_language,
    respond,
    route_to_flow,
    split_user_request,
    synthesize,
    translate,
)
from .prompts import (
    FIND_RELEVANT,
    FIND_SIMILAR,
)
from .resources import (
    FAQ,
    GENEALOGY,
    RECORDS,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.graph import CompiledGraph


class ArkeyState(TypedDict):
    """The state of the workflow."""

    original: str
    "The user's original request expressed in his/her own language."
    language: str
    "The user's language."
    english: str
    "A translation of the user's request, in english."
    qid: int
    "Identifier of the question currently being answered."
    questions: list[str]
    "A breakdown of the user's original requests."
    sources: Annotated[list[str], add]
    "The list of sources that have been used to anwer a partial question."
    partial: Annotated[list[str], add]
    "A list of partial answers to the original user's request."
    en_response: str
    "A response to the overall user's request (in English)."
    response: str
    "The user facing response."


async def determine_language(state: ArkeyState) -> dict[str, str]:
    """Return the user's language."""
    lang = await find_language.ainvoke(state["original"])
    return {"language": lang}

async def to_english(state: ArkeyState) -> dict[str, str]:
    """Translate the user's message to English."""
    trans = await translate.ainvoke({"input": state["original"], "target": "english"})
    return {"english": trans}

async def split(state: ArkeyState) -> dict[str, list[str]|int ]:
    """Split the user's request."""
    questions = await split_user_request.ainvoke(state["english"])
    return {"questions": questions.questions, "qid": 0}

async def route(state: ArkeyState) -> Literal["faq", "genealogy", "records", "synthesis"]:
    """Route the question towards the most appropriate flow."""
    qid = state["qid"]
    questions = state["questions"]
    if qid < len(questions):
        question = questions[qid]
        flow = await route_to_flow.ainvoke(question)
        match x:=flow.lower():
            case _ if "genealogy" in x:
                return "genealogy"
            case _ if "records" in x:
                return "records"
            case _:
                return "faq"
    return "synthesis"

async def faq(state: ArkeyState) -> dict[str, str]:
    """Search in the faq."""
    qid = state["qid"]
    question = state["questions"][qid]
    retriever= FAQ.as_retriever(search_type="mmr")
    prompt = FIND_SIMILAR.format(input=question)
    sources = await retriever.ainvoke(prompt)
    sources = [e.metadata["fulltext"] for e in sources ]
    sources = "\n\n".join( sources )
    response= await respond.ainvoke({"question": question, "facts": sources})
    return {"partial": [response], "sources": [sources], "qid": qid + 1}

async def genealogy(state: ArkeyState) -> dict[str, str]:
    """Search in the genealogic sources."""
    qid = state["qid"]
    question = state["questions"][qid]
    retriever= GENEALOGY.as_retriever(search_type="mmr")
    prompt = FIND_RELEVANT.format(input=question)
    sources = await retriever.ainvoke(prompt)
    sources = [e.page_content for e in sources ]
    sources = "\n\n".join( sources )
    response= await respond.ainvoke({"question": question, "facts": sources})
    return {"partial": [response], "sources": [sources], "qid": qid + 1}

async def records(state: ArkeyState) -> dict[str, str]:
    """Search in the records analysis sources."""
    qid = state["qid"]
    question = state["questions"][qid]
    retriever= RECORDS.as_retriever(search_type="mmr")
    prompt = FIND_RELEVANT.format(input=question)
    sources = await retriever.ainvoke(prompt)
    sources = [e.page_content for e in sources ]
    sources = "\n\n".join( sources )
    response= await respond.ainvoke({"question": question, "facts": sources})
    return {"partial": [response], "sources": [sources], "qid": qid + 1}

async def synthesis(state: ArkeyState) -> dict[str, str]:
    """Synthesize the partial responses made along the various flows."""
    partials = state["partial"]
    partials = "\n\n".join(partials)
    response = await synthesize.ainvoke({"original": state["english"], "partials": partials})
    return {"en_response": response}

async def to_user_language(state: ArkeyState) -> dict[str, str]:
    """Translate the synthetic response back to the user's language."""
    text = state["en_response"]
    target= state["language"]
    trans = await translate.ainvoke({"input": text, "target": target})
    return {"response": trans}

def arkey(checkpointer: BaseCheckpointSaver = None) -> CompiledGraph:
    """Create a workflow for the arkeybot."""
    _builder = StateGraph(ArkeyState)
    _builder.add_node("determine_language", determine_language)
    _builder.add_node("to_english", to_english)
    _builder.add_node("split", split)
    _builder.add_node("faq", faq)
    _builder.add_node("genealogy", genealogy)
    _builder.add_node("records", records)
    _builder.add_node("synthesis", synthesis)
    _builder.add_node("to_user_language", to_user_language)

    _builder.add_edge(START, "determine_language")
    _builder.add_edge(START, "to_english")
    _builder.add_edge("to_english", "split")
    _builder.add_conditional_edges("split", route)
    _builder.add_conditional_edges("faq", route)
    _builder.add_conditional_edges("genealogy", route)
    _builder.add_conditional_edges("records", route)
    _builder.add_edge(["determine_language", "synthesis"], "to_user_language")
    _builder.add_edge("to_user_language", END)

    return _builder.compile(checkpointer)
