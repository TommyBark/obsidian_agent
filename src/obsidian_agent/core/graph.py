# obsidian_agent/core/graph.py
from langgraph.graph import START, StateGraph

import obsidian_agent.core.configuration as configuration
from obsidian_agent.core.models import GraphState
from obsidian_agent.core.nodes.assistant import obsidian_assistant_node
from obsidian_agent.core.nodes.notes import (
    create_note_node,
    read_notes_node,
    search_notes_node,
)
from obsidian_agent.core.nodes.profile import (
    update_instructions_node,
    update_profile_node,
)
from obsidian_agent.core.nodes.router import route_message
from obsidian_agent.core.store import checkpoint_factory, store_factory


def create_graph():
    # Create the graph + all nodes
    builder = StateGraph(GraphState, config_schema=configuration.Configuration)

    # Add nodes
    builder.add_node("obsidian_assistant", obsidian_assistant_node)
    builder.add_node("create_note", create_note_node)
    builder.add_node("read_notes", read_notes_node)
    builder.add_node("update_profile", update_profile_node)
    builder.add_node("update_instructions", update_instructions_node)
    builder.add_node("search_notes", search_notes_node)

    # Add edges
    builder.add_edge(START, "obsidian_assistant")
    builder.add_conditional_edges("obsidian_assistant", route_message)
    builder.add_edge("create_note", "obsidian_assistant")
    builder.add_edge("read_notes", "obsidian_assistant")
    builder.add_edge("update_profile", "obsidian_assistant")
    builder.add_edge("update_instructions", "obsidian_assistant")
    builder.add_edge("search_notes", "obsidian_assistant")

    # Configure memory
    across_thread_memory = store_factory("memory")
    within_thread_memory = checkpoint_factory("memory")

    return builder.compile(
        checkpointer=within_thread_memory, store=across_thread_memory
    )


# Create the graph instance
graph = create_graph()
