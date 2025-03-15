import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.store.memory import InMemoryStore

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


class MockPath:
    def __init__(self, name):
        self.name = name


@pytest.fixture
def mock_store():
    """Create a mock store with memory capabilities."""
    store = InMemoryStore()
    return store


@pytest.fixture
def mock_config():
    """Create a mock config with user_id."""
    return {"configurable": {"user_id": "test-user", "recursion_limit": 10}}


@pytest.fixture
def mock_obsidian_library():
    """Create a mock ObsidianLibrary with controlled behavior."""
    with patch("obsidian_agent.core.nodes.notes.LIBRARY") as mock_library:
        # Mock search_notes
        mock_library.search_notes.return_value = [
            Document(
                metadata={"path": MockPath(name="Note1.md")},
                page_content="Content of Note1",
            ),
            Document(
                metadata={"path": MockPath(name="Note2.md")},
                page_content="Content of Note2",
            ),
        ]

        # Mock get_note_with_context
        mock_library.get_note_with_context.return_value = (
            "# Note Title\n\nNote content with [[links]]"
        )

        # Mock put_note
        mock_library.put_note.return_value = None

        yield mock_library


@pytest.fixture
def update_profile_state():
    """Create a state with messages for updating user profile."""
    return {
        "messages": [
            HumanMessage(content="My name is John and I live in New York."),
            AIMessage(content="Nice to meet you, John!"),
            HumanMessage(content="I work as a software engineer."),
            HumanMessage(
                content="",
                tool_calls=[{"id": "call_123", "name": "UpdateProfile", "args": {}}],
            ),
        ]
    }


@pytest.fixture
def update_instructions_state():
    """Create a state with messages about note creation preferences."""
    return {
        "messages": [
            HumanMessage(content="When creating notes, please include a date header."),
            AIMessage(content="I'll make sure to include date headers in new notes."),
            HumanMessage(content="Also add a summary section at the end."),
            HumanMessage(
                content="",
                tool_calls=[
                    {"id": "call_456", "name": "UpdateInstructions", "args": {}}
                ],
            ),
        ]
    }


@pytest.fixture
def search_notes_state():
    """Create a state with a search request."""
    return {
        "messages": [
            HumanMessage(content="Search for notes about python"),
            HumanMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_789",
                        "name": "SearchNotes",
                        "args": {"keywords": "python", "k": 2},
                    }
                ],
            ),
        ]
    }


@pytest.fixture
def read_notes_state():
    """Create a state with a read request."""
    return {
        "messages": [
            HumanMessage(content="Read the Python note"),
            HumanMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_101",
                        "name": "ReadNote",
                        "args": {"note_name": "Python", "depth": 1},
                    }
                ],
            ),
        ]
    }


@pytest.fixture
def create_notes_state():
    """Create a state with a create request."""
    return {
        "messages": [
            HumanMessage(content="Create a note about Python"),
            HumanMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_202",
                        "name": "CreateNote",
                        "args": {
                            "note_name": "Python",
                            "note_text": "# Python\n\nPython is a programming language.",
                        },
                    }
                ],
            ),
        ]
    }


@pytest.fixture
def update_memory_user_state():
    """Create a state with UpdateMemory request for user profile."""
    return {
        "messages": [
            HumanMessage(content="My name is John and I work as a developer."),
            HumanMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_303",
                        "name": "UpdateMemory",
                        "args": {"update_type": "user"},
                    }
                ],
            ),
        ]
    }


@pytest.fixture
def update_memory_instructions_state():
    """Create a state with UpdateMemory request for instructions."""
    return {
        "messages": [
            HumanMessage(content="When creating notes, add tags at the top."),
            HumanMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_404",
                        "name": "UpdateMemory",
                        "args": {"update_type": "instructions"},
                    }
                ],
            ),
        ]
    }


@pytest.fixture
def setup_store_with_profile_and_instructions(mock_store):
    """Setup store with profile and instructions."""
    user_id = "test-user"
    profile_namespace = ("profile", user_id)
    instructions_namespace = ("instructions", user_id)

    # Add profile to store
    profile_value = {
        "name": "John",
        "location": "New York",
        "job": "software engineer",
        "interests": ["programming", "reading"],
        "connections": [],
    }
    mock_store.put(profile_namespace, "profile_key", profile_value)

    # Add instructions to store
    instructions_value = {"memory": "Include date headers and categorize all notes."}
    mock_store.put(instructions_namespace, "instructions_key", instructions_value)

    return mock_store
