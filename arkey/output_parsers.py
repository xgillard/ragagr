"""Output parsers."""
import re


def reasoning_answer_only(text: str) -> str:
    """Parse the finaloutput of a reasoning LLM call."""
    match = re.search(r"</think>\s*(.*?)\s*$", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()
