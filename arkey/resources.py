"""All the resources that can be used by other parts of the graph."""

import torch
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

__DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__DIGITA = "/home/xgillard/REPO/rag/agr/notebooks/digita"

# The chat model used for the 'smart' interactions
CHAT  = ChatOllama(base_url="http://localhost:11434", model="gemma3:4b", temperature=0.1, verbose=True)
THINK = ChatOllama(base_url="http://localhost:11434", model="deepseek-r1:8b", temperature=0.1, verbose=True)

# The FAQ database
FAQ = Chroma(
    "faq",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory=__DIGITA,
)

# The Getting started with Genealogy database
GENEALOGY = Chroma(
    "genealogy",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory=__DIGITA,
)

# The Getting started with Genealogy database
RECORDS = Chroma(
    "records",
    HuggingFaceEmbeddings(
        model_name   = "intfloat/multilingual-e5-large-instruct",
        model_kwargs = {"device": __DEVICE, "model_kwargs": {"torch_dtype": "float16"}},
    ),
    persist_directory=__DIGITA,
)
