import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from obsidian_agent.core.nodes.assistant import obsidian_assistant_node


def test_obsidian_assistant_node_with_profile_and_instructions(
    mock_config, setup_store_with_profile_and_instructions
):
    """Test that assistant node correctly uses profile and instructions in the system message."""
    # Create mock state
    mock_state = {"messages": [HumanMessage(content="Hello")]}

    # Mock the model response
    with patch("obsidian_agent.core.nodes.assistant.model") as mock_model:
        mock_model_instance = MagicMock()
        mock_model.bind_tools.return_value = mock_model_instance
        mock_model_instance.invoke.return_value = {
            "role": "assistant",
            "content": "Hello John! How can I help you today?",
        }

        # Execute the node
        result = obsidian_assistant_node(
            mock_state, mock_config, setup_store_with_profile_and_instructions
        )

    # Get the system message from the model invocation
    call_args = mock_model_instance.invoke.call_args
    messages = call_args[0][0]
    system_message = messages[0]

    # Verify system message contains profile and instructions
    assert "John" in system_message.content
    assert "New York" in system_message.content
    assert "software engineer" in system_message.content
    assert "programming" in system_message.content
    assert "Include date headers" in system_message.content
    assert "categorize all notes" in system_message.content

    # Check that response is passed through
    assert "messages" in result
    assert result["messages"][0]["content"] == "Hello John! How can I help you today?"


def test_obsidian_assistant_node_without_profile_or_instructions(
    mock_config, mock_store
):
    """Test that assistant node works correctly without profile or instructions."""
    # Create mock state
    mock_state = {"messages": [HumanMessage(content="Hello")]}

    # Mock the model response
    with patch("obsidian_agent.core.nodes.assistant.model") as mock_model:
        mock_model_instance = MagicMock()
        mock_model.bind_tools.return_value = mock_model_instance
        mock_model_instance.invoke.return_value = {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
        }

        # Execute the node
        result = obsidian_assistant_node(mock_state, mock_config, mock_store)

    # Get the system message from the model invocation
    call_args = mock_model_instance.invoke.call_args
    messages = call_args[0][0]
    system_message = messages[0]

    # Verify system message doesn't contain profile or instructions
    assert "user_profile" in system_message.content
    assert "instructions" in system_message.content

    # Verify that None is shown for missing profile and instructions are empty
    assert "<user_profile>\nNone\n</user_profile>" in system_message.content

    # Check that response is passed through
    assert "messages" in result
    assert result["messages"][0]["content"] == "Hello! How can I help you today?"


def test_obsidian_assistant_node_with_custom_assistant_role(mock_config, mock_store):
    """Test that assistant node uses custom assistant role from config."""
    # Create mock state
    mock_state = {"messages": [HumanMessage(content="Hello")]}

    # Extend config with custom assistant role
    custom_config = mock_config.copy()
    custom_config["configurable"][
        "assistant_role"
    ] = "You are a specialized Obsidian assistant focused on academic research."

    # Mock the model response
    with patch("obsidian_agent.core.nodes.assistant.model") as mock_model:
        mock_model_instance = MagicMock()
        mock_model.bind_tools.return_value = mock_model_instance
        mock_model_instance.invoke.return_value = {
            "role": "assistant",
            "content": "Hello! How can I help with your research?",
        }

        # Execute the node
        result = obsidian_assistant_node(mock_state, custom_config, mock_store)

    # Get the system message from the model invocation
    call_args = mock_model_instance.invoke.call_args
    messages = call_args[0][0]
    system_message = messages[0]

    # Verify system message includes custom assistant role
    assert "specialized Obsidian assistant" in system_message.content
    assert "academic research" in system_message.content

    # Check that response is passed through
    assert "messages" in result
    assert (
        result["messages"][0]["content"] == "Hello! How can I help with your research?"
    )


def test_obsidian_assistant_node_tool_binding(mock_config, mock_store):
    """Test that assistant node correctly binds tools to the model."""
    # Create mock state
    mock_state = {"messages": [HumanMessage(content="Hello")]}

    # Mock the model response
    with patch("obsidian_agent.core.nodes.assistant.model") as mock_model:
        mock_model_instance = MagicMock()
        mock_model.bind_tools.return_value = mock_model_instance
        mock_model_instance.invoke.return_value = {
            "role": "assistant",
            "content": "Hello!",
        }

        # Execute the node
        result = obsidian_assistant_node(mock_state, mock_config, mock_store)

    # Verify that bind_tools was called with the correct tools
    bind_tools_args = mock_model.bind_tools.call_args[1]
    assert "tools" in bind_tools_args
    assert len(bind_tools_args["tools"]) == 4  # Should have 4 tools

    # Verify tool names
    tool_names = [tool.__name__ for tool in bind_tools_args["tools"]]
    assert "UpdateMemory" in tool_names
    assert "CreateNote" in tool_names
    assert "ReadNote" in tool_names
    assert "SearchNotes" in tool_names

    # Check that tool_choice is set to "auto"
    assert bind_tools_args["tool_choice"] == "auto"
