import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from obsidian_agent.core.nodes.tools import tools_node


def test_tools_node_search_notes_routing(mock_config, mock_store):
    """Test that tools_node routes SearchNotes to search_notes_node."""
    # Create a state with a SearchNotes tool call
    search_state = {
        "messages": [
            HumanMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_search",
                        "name": "SearchNotes",
                        "args": {"keywords": "python"},
                    }
                ],
            )
        ]
    }

    # Mock the nodes
    with patch("obsidian_agent.core.nodes.tools.search_notes_node") as mock_search:
        mock_search.return_value = {
            "messages": [{"role": "tool", "content": "search result"}]
        }

        # Execute the node
        result = tools_node(search_state, mock_config, mock_store)

    # Verify search_notes_node was called
    mock_search.assert_called_once()

    # Check that the result is from search_notes_node
    assert "messages" in result
    assert result["messages"][0]["content"] == "search result"


def test_tools_node_read_notes_routing(mock_config, mock_store):
    """Test that tools_node routes ReadNote to read_notes_node."""
    # Create a state with a ReadNote tool call
    read_state = {
        "messages": [
            HumanMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_read",
                        "name": "ReadNote",
                        "args": {"note_name": "Python"},
                    }
                ],
            )
        ]
    }

    # Mock the nodes
    with patch("obsidian_agent.core.nodes.tools.read_notes_node") as mock_read:
        mock_read.return_value = {
            "messages": [{"role": "tool", "content": "read result"}]
        }

        # Execute the node
        result = tools_node(read_state, mock_config, mock_store)

    # Verify read_notes_node was called
    mock_read.assert_called_once()

    # Check that the result is from read_notes_node
    assert "messages" in result
    assert result["messages"][0]["content"] == "read result"


# def test_tools_node_create_note_routing(mock_config, mock_store):
#     """Test that tools_node routes CreateNote to create_note_node."""
#     # Create a state with a CreateNote tool call
#     create_state = {
#         "messages": [
#             HumanMessage(
#                 content="",
#                 tool_calls=[
#                     {
#                         "id": "call_create",
#                         "name": "CreateNote",
#                         "args": {
#                             "note_name": "Python",
#                             "note_text": "# Python\n\nPython is a programming language.",
#                         },
#                     }
#                 ],
#             )
#         ]
#     }

#     # Mock the nodes
#     with patch("obsidian_agent.core.nodes.tools.create_note_node") as mock_create:
#         mock_create.return_value = {
#             "messages": [{"role": "tool", "content": "create result"}]
#         }

#         # Execute the node
#         result = tools_node(create_state, mock_config, mock_store)

#     # Verify create_note_node was called
#     mock_create.assert_called_once()

#     # Check that the result
