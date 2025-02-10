import os

import requests

from obsidian_agent.core.environment import LIBRARY
from obsidian_agent.core.models import Note

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
