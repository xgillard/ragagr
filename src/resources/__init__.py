"""All the resources that can be used by other parts of the graph."""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# The chat model used for the 'smart' interactions
CHAT = ChatOllama(base_url="http://localhost:11434", model="gemma3:4b", temperature=0)

# The FAQ database
FAQ = Chroma(
    "faq",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": "cuda", "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory="./digita",
)
