import os
from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from obsidian_utils import ObsidianLibrary


def create_vector_store(obsidian_path: str, store_path: Optional[str] = None) -> FAISS:
    """
    Creates a FAISS vector store from a list of documents.

    Args:
        docs (List[Document]): A list of Document objects containing the content to be stored.
        store_path (Optional[str]): The path to store the vector store locally. If None, the vector store will not be stored.

    Returns:
        FAISS: The FAISS vector store containing the documents.
    """
    embedding_model = OpenAIEmbeddings()

    if store_path:
        if Path(store_path).exists():
            print("Loading existing store")
            return FAISS.load_local(store_path, embedding_model)

    file_paths = [*Path(obsidian_path).rglob("*.md")]
    docs = []
    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            docs.append(Document(page_content=text, metadata={"path": path}))

    # Creating text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7500,
        chunk_overlap=200,
    )

    texts = text_splitter.split_documents(docs)

    # Create the FAISS vector store
    store = FAISS.from_documents(texts, embedding_model)

    # Save the vector store locally if a path is provided
    if store_path:
        store.save_local(store_path)
        print(f"Store saved to {store_path}")
    return store


if __name__ == "__main__":
    OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
    if not OBSIDIAN_VAULT_PATH:
        raise ValueError("OBSIDIAN_VAULT_PATH environment variable not set.")
    obsidian = ObsidianLibrary(OBSIDIAN_VAULT_PATH)

    t = create_vector_store(OBSIDIAN_VAULT_PATH, store_path="./test_store")
    # print(len(filenames))
    # store = create_vector_store(docs, store_path="test_store")
    # print(store)
