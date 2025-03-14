import os
import uuid
from datetime import datetime
from typing import Literal

import requests
from langchain_core.messages import HumanMessage, SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.store.base import BaseStore

import obsidian_agent.core.configuration as configuration
from obsidian_agent.core.environment import LIBRARY, model
from obsidian_agent.core.models import GraphState, Note
from obsidian_agent.core.nodes.profile import (
    CREATE_INSTRUCTIONS,
    TRUSTCALL_INSTRUCTION,
    profile_extractor,
)

# JINA_API_KEY = os.getenv("JINA_API_KEY")
# if JINA_API_KEY is None:
#     raise ValueError("Please set the JINA_API_KEY environment variable.")

def scrape_page_jina(url:str) -> str:
    """
    Scrapes content of a webpage using JinaAI.

    Args:
        url (str): The URL of the webpage to scrape.
    
    Returns:
        str: The scraped markdown content of the webpage.
    """
    #headers = {"Authorization":"Bearer " + JINA_API_KEY}
    response = requests.get("https://r.jina.ai" + url)

    markdown_content = response.text

    return markdown_content

@tool
def search_notes(keywords: str, k: int = 5) -> str:
    """
    Search notes in the vector store based on keywords.
    
    Args:
        keywords (str): The keywords to search for.
        k (int): The number of results to return.

    Returns:
        str: Formatted string of note names and texts.
    """
    results = LIBRARY.search_notes(keywords, k)
    content = [
        Note(name=doc.metadata["path"].name, text=doc.page_content) for doc in results
    ]

    updated_content = "\n---------------".join(
        [f"NOTENAME: {note.name}\n {note.text}" for note in content]
    )

    return updated_content

@tool
def create_note(note_name: str, note_text: str) -> str:
    """
    Creates a note in the library with the given name and text.
    
    Args:
        note_name (str): The name of the note to be created.
        note_text (str): The content of the note to be created.
        
    Returns:
        str: A message indicating the result of the creation attempt.
    """
    if not note_name or not note_text:
        return "Error: Note name and text cannot be empty."
        
    try:
        LIBRARY.put_note(note_name, note_text)
        content = f"Note: {note_name} has been created."
    except FileExistsError as e:
        content = str(e)
    except Exception as e:
        content = f"An unexpected error occurred: {str(e)}"

    return content

@tool
def read_notes_with_context(note_name:str, depth:int=0) -> str:
    """
    Reads a note and recursively also its linked notes with specified depth.

    Args:
        note_name (str): The name of the note to read.
        depth (int): The depth of linked notes to read.

    Returns:
        str: The content of the note and its linked notes.
    """
    try:
        content = LIBRARY.get_note_with_context(note_name, depth)
    except (ValueError, FileNotFoundError) as e:
        content = str(e)

    return content


def update_profile(state: GraphState, config: RunnableConfig, store: BaseStore) -> str:
    """
    Update the user profile based on chat history.
    
    Args:
        state: Current graph state containing messages
        config: Configuration for the runnable
        store: Storage for persisting profile data
        
    Returns:
        str: Message indicating profile was updated
    """
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace = ("profile", user_id)
    
    # Retrieve existing memories for context
    existing_items = store.search(namespace)
    existing_memories = (
        [
            (existing_item.key, "Profile", existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Prepare messages for extraction
    instruction = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=instruction)] + state["messages"][:-1]
        )
    )

    # Extract and save memories
    result = profile_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )
    
    # Save extracted memories
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )
    
    return "Profile has been updated based on our conversation."

def update_instructions(state: GraphState, config: RunnableConfig, store: BaseStore) -> str:
    """
    Update custom instructions based on chat history.
    
    Args:
        state: Current graph state containing messages
        config: Configuration for the runnable
        store: Storage for persisting instructions
        
    Returns:
        str: Message indicating instructions were updated
    """
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    namespace = ("instructions", user_id)
    
    # Get existing instructions
    existing_memory = store.get(namespace, "user_instructions")
    
    # Format system message
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    
    # Generate new instructions
    new_memory = model.invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [HumanMessage(content="Please update the instructions based on the conversation")]
    )
    
    # Save new instructions
    store.put(namespace, "user_instructions", {"memory": new_memory.content})
    
    return "Instructions have been updated based on our conversation."

# Then we can create a tool that combines both:
@tool
def update_memory(state: GraphState, config: RunnableConfig, store: BaseStore, update_type: Literal["user", "instructions"]) -> str:
    """
    Update either user profile or custom instructions.
    
    Args:
        state: Current graph state
        config: Runnable configuration
        store: Storage backend
        update_type: Type of update - either "user" for profile or "instructions" for custom instructions
        
    Returns:
        str: Message indicating what was updated
    """
    if update_type == "user":
        return update_profile(state, config, store)
    elif update_type == "instructions":
        return update_instructions(state, config, store)
    else:
        raise ValueError(f"Unknown update_type: {update_type}")