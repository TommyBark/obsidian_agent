import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from obsidian_agent.core.nodes.notes import (
    create_note_node,
    read_notes_node,
    search_notes_node,
)


def test_search_notes_node(
    search_notes_state, mock_config, mock_store, mock_obsidian_library
):
    """Test that search_notes_node correctly searches and formats results."""
    # Setup mock documents with metadata
    doc1 = Document(
        page_content="Content of Note1", metadata={"path": Path(name="Note1.md")}
    )
    doc2 = Document(
        page_content="Content of Note2", metadata={"path": Path(name="Note2.md")}
    )
    mock_obsidian_library.search_notes.return_value = [doc1, doc2]

    pytest.set_trace()
    # Execute the node
    result = search_notes_node(search_notes_state, mock_config, mock_store)

    # Verify search was called with correct parameters
    mock_obsidian_library.search_notes.assert_called_once_with("python", 2)

    # Check that the response contains formatted results
    assert "messages" in result
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_789"
    assert "NOTENAME: Note1.md" in result["messages"][0]["content"]
    assert "Content of Note1" in result["messages"][0]["content"]
    assert "NOTENAME: Note2.md" in result["messages"][0]["content"]
    assert "Content of Note2" in result["messages"][0]["content"]


def test_read_notes_node(
    read_notes_state, mock_config, mock_store, mock_obsidian_library
):
    """Test that read_notes_node correctly fetches and returns note content."""
    # Setup mock note content
    expected_content = (
        "# Python\n\nPython is a programming language with [[links]] to other notes."
    )
    mock_obsidian_library.get_note_with_context.return_value = expected_content

    # Execute the node
    result = read_notes_node(read_notes_state, mock_config, mock_store)

    # Verify get_note_with_context was called with correct parameters
    mock_obsidian_library.get_note_with_context.assert_called_once_with("Python", 1)

    # Check that the response contains the note content
    assert "messages" in result
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_101"
    assert result["messages"][0]["content"] == expected_content


def test_read_notes_node_error_handling(
    read_notes_state, mock_config, mock_store, mock_obsidian_library
):
    """Test that read_notes_node handles errors correctly."""
    # Setup mock to raise an error
    mock_obsidian_library.get_note_with_context.side_effect = FileNotFoundError(
        "Note 'Python' not found"
    )

    # Execute the node
    result = read_notes_node(read_notes_state, mock_config, mock_store)

    # Check that the error is properly returned
    assert "messages" in result
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_101"
    assert "Note 'Python' not found" in result["messages"][0]["content"]


def test_create_note_node(
    create_notes_state, mock_config, mock_store, mock_obsidian_library
):
    """Test that create_note_node correctly creates a note."""
    # Execute the node
    result = create_note_node(create_notes_state, mock_config, mock_store)

    # Verify put_note was called with correct parameters
    mock_obsidian_library.put_note.assert_called_once_with(
        "Python", "# Python\n\nPython is a programming language."
    )

    # Check that the response confirms note creation
    assert "messages" in result
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_202"
    assert "has been created" in result["messages"][0]["content"]


def test_create_note_node_handles_existing_note(
    create_notes_state, mock_config, mock_store, mock_obsidian_library
):
    """Test that create_note_node handles the case when a note already exists."""
    # Setup mock to raise FileExistsError
    mock_obsidian_library.put_note.side_effect = FileExistsError(
        "Note 'Python' already exists"
    )

    # Execute the node
    result = create_note_node(create_notes_state, mock_config, mock_store)

    # Check that the error is properly returned
    assert "messages" in result
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_202"
    assert "already exists" in result["messages"][0]["content"]


def test_search_notes_node_with_custom_k(
    mock_config, mock_store, mock_obsidian_library
):
    """Test search_notes_node with a custom number of results."""
    # Create a state with a search request with custom k
    custom_k_state = {
        "messages": [
            {
                "role": "user",
                "content": "Search for notes about python, show me 10 results",
            },
            {
                "role": "user",
                "tool_calls": [
                    {
                        "id": "call_custom",
                        "name": "SearchNotes",
                        "args": {"keywords": "python", "k": 10},
                    }
                ],
            },
        ]
    }

    # Execute the node
    result = search_notes_node(custom_k_state, mock_config, mock_store)

    # Verify search was called with custom k
    mock_obsidian_library.search_notes.assert_called_once_with("python", 10)
