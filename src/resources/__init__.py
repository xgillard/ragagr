"""All the resources that can be used by other parts of the graph."""

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

__DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# The chat model used for the 'smart' interactions
CHAT = ChatOllama(base_url="http://localhost:11434", model="gemma3:4b", temperature=0)

# The FAQ database
FAQ = Chroma(
    "faq",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory="./digita",
)

# The Getting started with Genealogy database
GENEALOGY = Chroma(
    "genealogy",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory="./digita",
)
