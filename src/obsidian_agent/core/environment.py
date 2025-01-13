import os

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
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
