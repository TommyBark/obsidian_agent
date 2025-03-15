from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from obsidian_agent.core.models import GraphState
from obsidian_agent.core.tools import scrape_page_jina


def get_url_content_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    tool_call = state["messages"][-1].tool_calls[0]
    url = tool_call["args"]["url"]
    content = scrape_page_jina(url)

    return {
        "messages": [
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call["id"],
            }
        ]
    }
