import os
import sys
from unittest.mock import MagicMock

import pytest

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from obsidian_agent.utils.common import extract_tool_info


def test_extract_tool_info_for_patch_doc():
    """Test that extract_tool_info correctly processes PatchDoc tool calls."""
    # Define tool calls with PatchDoc
    tool_calls = [
        [
            {
                "name": "PatchDoc",
                "args": {
                    "json_doc_id": "123",
                    "planned_edits": "Update user's location",
                    "patches": [{"value": "New York"}],
                },
            }
        ]
    ]

    # Extract tool info
    result = extract_tool_info(tool_calls)

    # Verify the result
    assert "Document 123 updated" in result
    assert "Plan: Update user's location" in result
    assert "Added content: New York" in result


def test_extract_tool_info_for_memory():
    """Test that extract_tool_info correctly processes Memory tool calls."""
    # Define tool calls with Memory
    tool_calls = [
        [
            {
                "name": "Memory",
                "args": {
                    "name": "John",
                    "location": "San Francisco",
                    "job": "Software Engineer",
                },
            }
        ]
    ]

    # Extract tool info
    result = extract_tool_info(tool_calls)

    # Verify the result
    assert "New Memory created" in result
    assert "Content: " in result
    assert "name" in result
    assert "John" in result
    assert "location" in result
    assert "San Francisco" in result
    assert "job" in result
    assert "Software Engineer" in result


def test_extract_tool_info_for_custom_schema():
    """Test that extract_tool_info correctly processes custom schema tool calls."""
    # Define tool calls with a custom schema
    tool_calls = [
        [
            {
                "name": "ToDo",
                "args": {
                    "task": "Buy groceries",
                    "priority": "High",
                    "due_date": "2023-01-01",
                },
            }
        ]
    ]

    # Extract tool info with custom schema
    result = extract_tool_info(tool_calls, schema_name="ToDo")

    # Verify the result
    assert "New ToDo created" in result
    assert "Content: " in result
    assert "task" in result
    assert "Buy groceries" in result
    assert "priority" in result
    assert "High" in result
    assert "due_date" in result
    assert "2023-01-01" in result


def test_extract_tool_info_mixed():
    """Test that extract_tool_info correctly processes mixed tool calls."""
    # Define tool calls with mixed types
    tool_calls = [
        [
            {
                "name": "PatchDoc",
                "args": {
                    "json_doc_id": "123",
                    "planned_edits": "Update job",
                    "patches": [{"value": "Data Scientist"}],
                },
            }
        ],
        [{"name": "Memory", "args": {"name": "Jane", "location": "London"}}],
    ]

    # Extract tool info
    result = extract_tool_info(tool_calls)

    # Verify the result contains both updates
    assert "Document 123 updated" in result
    assert "Plan: Update job" in result
    assert "Added content: Data Scientist" in result
    assert "New Memory created" in result
    assert "Content: " in result
    assert "name" in result
    assert "Jane" in result
    assert "location" in result
    assert "London" in result
