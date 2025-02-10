import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from obsidian_agent.utils.obsidian import ObsidianLibrary


def initialize_environment():
    """Initialize environment variables and library"""
    OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")

    if OBSIDIAN_VAULT_PATH is None:
        raise ValueError("Please set the OBSIDIAN_VAULT_PATH environment variable.")
    if VECTOR_STORE_PATH is None:
        raise ValueError("Please set the VECTOR_STORE_PATH environment variable.")

    return ObsidianLibrary(
        path=OBSIDIAN_VAULT_PATH, vector_store_path=VECTOR_STORE_PATH
    )


LIBRARY = initialize_environment()
model_name = os.getenv("MODEL_NAME")
if model_name is None:
    raise ValueError("Please set the MODEL_NAME environment variable.")
elif model_name == "gpt-4o-mini":
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
elif model_name == "gemini-2.0-flash":
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
else:
    raise ValueError(f"Unknown model name: {model_name}")
