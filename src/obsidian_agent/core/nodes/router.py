from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.store.base import BaseStore

from obsidian_agent.core.models import GraphState


def route_message(
    state: GraphState, config: RunnableConfig, store: BaseStore
) -> Literal[
    END,
    "create_note",
    "update_instructions",
    "update_profile",
    "read_notes",
    "search_notes",
]:
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END

    tool_call = message.tool_calls[0]
    if tool_call["args"].get("update_type", None) is not None:
        if tool_call["args"]["update_type"] == "user":
            return "update_profile"
        elif tool_call["args"]["update_type"] == "new_note":
            return "create_note"
        elif tool_call["args"]["update_type"] == "instructions":
            return "update_instructions"
        else:
            raise ValueError
    elif tool_call["args"].get("note_name", None) is not None:
        return "read_notes"
    elif tool_call["args"].get("keywords", None) is not None:
        return "search_notes"
    else:
        raise ValueError
