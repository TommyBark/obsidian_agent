# obsidian_agent/core/graph.py
from langgraph.graph import START, StateGraph

import obsidian_agent.core.configuration as configuration
from obsidian_agent.core.models import GraphState
from obsidian_agent.core.nodes.assistant import obsidian_assistant_node
from obsidian_agent.core.nodes.router import route_message
from obsidian_agent.core.nodes.tools import tools_node
from obsidian_agent.core.store import checkpoint_factory, store_factory


def create_graph():
    # Create the graph + all nodes
    builder = StateGraph(GraphState, config_schema=configuration.Configuration)

    # Add nodes
    builder.add_node("obsidian_assistant", obsidian_assistant_node)
    builder.add_node("tools", tools_node)

    # Add edges
    builder.add_edge(START, "obsidian_assistant")
    builder.add_conditional_edges("obsidian_assistant", route_message)
    builder.add_edge("tools", "obsidian_assistant")

    # Configure memory
    across_thread_memory = store_factory("memory")
    within_thread_memory = checkpoint_factory("memory")

    return builder.compile(
        checkpointer=within_thread_memory, store=across_thread_memory
    )


# Create the graph instance
graph = create_graph()
