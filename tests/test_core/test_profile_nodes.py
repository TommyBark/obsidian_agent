import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from obsidian_agent.core.nodes.profile import (
    update_instructions_node,
    update_profile_node,
)


def test_update_profile_node(update_profile_state, mock_config, mock_store):
    """Test that profile updates are correctly stored."""
    # Mock the profile_extractor.invoke method
    with patch("obsidian_agent.core.nodes.profile.profile_extractor") as mock_extractor:
        mock_extractor.invoke.return_value = {
            "responses": [
                MagicMock(
                    model_dump=MagicMock(
                        return_value={
                            "name": "John",
                            "location": "New York",
                            "job": "software engineer",
                            "interests": ["programming"],
                            "connections": [],
                        }
                    )
                )
            ],
            "response_metadata": [{"json_doc_id": "profile_123"}],
        }

        # Execute the node
        result = update_profile_node(update_profile_state, mock_config, mock_store)

    # Check that a tool response was returned
    assert "messages" in result
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_123"

    # Check that profile was stored in the store
    namespace = ("profile", "test-user")
    stored_profiles = mock_store.search(namespace)

    # Verify profile contains expected information
    assert len(stored_profiles) > 0
    profile = stored_profiles[0].value
    assert profile["name"] == "John"
    assert profile["location"] == "New York"
    assert profile["job"] == "software engineer"
    assert "programming" in profile["interests"]


def test_update_instructions_node(update_instructions_state, mock_config, mock_store):
    """Test that instruction updates are correctly stored."""
    # Execute the node
    with patch("obsidian_agent.core.nodes.profile.model") as mock_model:
        # Mock the model response for updating instructions
        mock_response = MagicMock()
        mock_response.content = (
            '{"memory": "Include date headers and summary sections in all new notes."}'
        )
        mock_model.invoke.return_value = mock_response

        result = update_instructions_node(
            update_instructions_state, mock_config, mock_store
        )

    # Check that a tool response was returned
    assert "messages" in result
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "call_456"

    # Check that instructions were stored in the store
    namespace = ("instructions", "test-user")
    stored_instructions = mock_store.search(namespace)

    # Verify instructions contain expected information
    assert len(stored_instructions) > 0
    instructions = stored_instructions[0].value
    assert "memory" in instructions
    assert "date headers" in instructions["memory"]
    assert "summary sections" in instructions["memory"]


def test_update_instructions_node_with_existing_instructions(
    update_instructions_state, mock_config, mock_store
):
    """Test that instruction updates correctly use existing instructions."""
    # Add existing instructions to the store
    user_id = "test-user"
    namespace = ("instructions", user_id)
    existing_value = {"memory": "Existing instructions about formatting"}
    mock_store.put(namespace, "user_instructions", existing_value)

    # Execute the node
    with patch("obsidian_agent.core.nodes.profile.model") as mock_model:
        # Capture the system message to verify it includes existing instructions
        calls = []

        def side_effect(messages, *args, **kwargs):
            calls.append(messages)
            mock_response = MagicMock()
            mock_response.content = (
                '{"memory": "Updated instructions with both old and new content"}'
            )
            return mock_response

        mock_model.invoke.side_effect = side_effect

        result = update_instructions_node(
            update_instructions_state, mock_config, mock_store
        )

    # Verify that the system message includes existing instructions
    system_message = calls[0][0].content
    assert "Existing instructions about formatting" in system_message

    # Check that updated instructions were stored in the store
    namespace = ("instructions", "test-user")
    stored_instructions = mock_store.search(namespace)
    assert len(stored_instructions) > 0
    assert "memory" in stored_instructions[0].value
    assert "Updated instructions" in stored_instructions[0].value["memory"]


def test_update_profile_node_with_existing_profile(
    update_profile_state, mock_config, mock_store
):
    """Test that profile updates correctly use existing profile information."""
    # Add existing profile to the store
    user_id = "test-user"
    namespace = ("profile", user_id)
    existing_value = {
        "name": "Old Name",
        "location": "Old Location",
        "job": None,
        "interests": ["old interest"],
        "connections": [],
    }
    mock_store.put(namespace, "existing_profile", existing_value)

    # Mock the profile_extractor.invoke method to verify it receives existing profile
    with patch("obsidian_agent.core.nodes.profile.profile_extractor") as mock_extractor:
        mock_extractor.invoke.return_value = {
            "responses": [
                MagicMock(
                    model_dump=MagicMock(
                        return_value={
                            "name": "John",
                            "location": "New York",
                            "job": "software engineer",
                            "interests": ["programming"],
                            "connections": [],
                        }
                    )
                )
            ],
            "response_metadata": [{"json_doc_id": "profile_123"}],
        }

        # Execute the node
        result = update_profile_node(update_profile_state, mock_config, mock_store)

        # Verify that existing memories were passed to the extractor
        args, kwargs = mock_extractor.invoke.call_args
        existing_memories = kwargs.get("existing") or args[0].get("existing")
        assert existing_memories is not None
        assert len(existing_memories) > 0
        assert existing_memories[0][0] == "existing_profile"
        assert existing_memories[0][1] == "Profile"
        assert existing_memories[0][2]["name"] == "Old Name"

    # Check that updated profile was stored in the store
    namespace = ("profile", "test-user")
    stored_profiles = mock_store.search(namespace)
    assert len(stored_profiles) > 0
