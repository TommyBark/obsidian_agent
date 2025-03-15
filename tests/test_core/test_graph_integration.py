import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph, END
from langgraph.store.memory import InMemoryStore

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from obsidian_agent.core.configuration import Configuration
from obsidian_agent.core.graph import create_graph
from obsidian_agent.core.models import GraphState
from obsidian_agent.core.nodes.router import route_message


def test_graph_creation():
    """Test that the graph is created correctly with all nodes and edges."""
    with patch("obsidian_agent.core.graph.StateGraph") as mock_state_graph:
        # Create a mock builder and mock compiled graph
        mock_builder = MagicMock()
        mock_compiled_graph = MagicMock()
        mock_state_graph.return_value = mock_builder
        mock_builder.compile.return_value = mock_compiled_graph

        # Create the graph
        graph = create_graph()

        # Verify that the builder was initialized correctly
        mock_state_graph.assert_called_once_with(
            GraphState, config_schema=Configuration
        )

        # Verify that nodes were added
        mock_builder.add_node.assert_any_call("obsidian_assistant", MagicMock())
        mock_builder.add_node.assert_any_call("tools", MagicMock())

        # Verify that edges were added
        mock_builder.add_edge.assert_any_call(START, "obsidian_assistant")
        mock_builder.add_edge.assert_any_call("tools", "obsidian_assistant")
        mock_builder.add_conditional_edges.assert_called_once()

        # Verify that the graph was compiled with memory and checkpoint
        mock_builder.compile.assert_called_once()
        assert mock_builder.compile.call_args[1]["checkpointer"] is not None
        assert mock_builder.compile.call_args[1]["store"] is not None

        # Verify that the return value is the compiled graph
        assert graph == mock_compiled_graph


@pytest.fixture
def simple_test_graph():
    """Create a simple test graph for integration testing."""
    # Mock dependencies
    with patch("obsidian_agent.core.nodes.assistant.model"):
        # Create a simple test graph
        builder = StateGraph(GraphState, config_schema=Configuration)

        # Add nodes
        builder.add_node(
            "obsidian_assistant",
            lambda state, config, store: {
                "messages": [AIMessage(content="Assistant response")]
            },
        )

        builder.add_node(
            "tools",
            lambda state, config, store: {
                "messages": [
                    {
                        "role": "tool",
                        "content": "Tool response",
                        "tool_call_id": "test_id",
                    }
                ]
            },
        )

        # Add edges
        builder.add_edge(START, "obsidian_assistant")
        builder.add_conditional_edges(
            "obsidian_assistant",
            lambda state, config, store: (
                "tools" if state["messages"][-1].tool_calls else END
            ),
        )
        builder.add_edge("tools", "obsidian_assistant")

        # Compile the graph
        graph = builder.compile(checkpointer=MemorySaver(), store=InMemoryStore())

        return graph


def test_graph_execution_without_tool_calls(simple_test_graph):
    """Test execution of the graph when no tool calls are made."""
    # Setup input without tool calls
    input_state = {"messages": [HumanMessage(content="Test message")]}

    # Execute the graph
    result = simple_test_graph.invoke(input_state)

    # Check the output
    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "Test message"
    assert result["messages"][1].content == "Assistant response"


def test_graph_execution_with_tool_calls(simple_test_graph):
    """Test execution of the graph when tool calls are made."""
    # Mock the conditional edge to always go to tools first
    with patch.object(
        simple_test_graph.nodes["obsidian_assistant"],
        "fn",
        return_value={
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[{"id": "test_id", "name": "TestTool", "args": {}}],
                )
            ]
        },
    ):
        # Setup input
        input_state = {"messages": [HumanMessage(content="Test message")]}

        # Execute the graph
        result = simple_test_graph.invoke(input_state)

        # Check the output - should have gone through both nodes and back to assistant
        assert len(result["messages"]) == 4
        assert result["messages"][0].content == "Test message"  # Initial message
        assert len(result["messages"][1].tool_calls) > 0  # Assistant with tool call
        assert result["messages"][2]["role"] == "tool"  # Tool response
        assert (
            result["messages"][3].content == "Assistant response"
        )  # Final assistant response


def test_route_message_to_tools():
    """Test that route_message correctly routes messages with tool calls to tools."""
    # Create a state with tool calls
    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[{"id": "test_id", "name": "TestTool", "args": {}}],
            )
        ]
    }

    # Execute the router
    result = route_message(state, {}, MagicMock())

    # Verify the result
    assert result == "tools"


def test_route_message_to_end():
    """Test that route_message correctly routes messages without tool calls to END."""
    # Create a state without tool calls
    state = {"messages": [AIMessage(content="No tool calls here")]}

    # Execute the router
    result = route_message(state, {}, MagicMock())

    # Verify the result
    assert result == END
