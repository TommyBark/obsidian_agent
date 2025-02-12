from typing import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.store.base import BaseStore

from obsidian_agent.core.models import GraphState


def route_message(
    state: GraphState, config: RunnableConfig, store: BaseStore
) -> Literal[END, "tools"]:
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END

    return "tools"