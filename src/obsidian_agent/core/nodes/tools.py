from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from obsidian_agent.core.models import GraphState
from obsidian_agent.core.nodes.notes import (
    create_note_node,
    read_notes_node,
    search_notes_node,
)
from obsidian_agent.core.nodes.profile import (
    update_instructions_node,
    update_profile_node,
)

from obsidian_agent.core.nodes.others import get_url_content_node

def tools_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    tool_call = state["messages"][-1].tool_calls[0]
    tool_name = tool_call["name"]

    # If UpdateMemory tool is called, we need to determine which tool to call
    if tool_name == "UpdateMemory":
        if tool_call["args"]["update_type"] == "user":
            tool_name = "UpdateProfile"
        elif tool_call["args"]["update_type"] == "instructions":
            tool_name = "UpdateInstructions"

    tool_map = {
        "SearchNotes": lambda: search_notes_node(state, config, store),
        "ReadNote": lambda: read_notes_node(state, config, store),
        "CreateNote": lambda: create_note_node(state, config, store),
        "UpdateProfile": lambda: update_profile_node(state, config, store),
        "UpdateInstructions": lambda: update_instructions_node(state, config, store),
        "GetURLContent": lambda: get_url_content_node(state, config, store),
    }

    if tool_name not in tool_map:
        raise ValueError(f"Unknown tool: {tool_name}")
        
    return tool_map[tool_name]()