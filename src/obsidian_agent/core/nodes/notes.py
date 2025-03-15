# obsidian_agent/core/nodes/notes.py
from datetime import datetime

from langchain_core.messages import SystemMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from trustcall import create_extractor

from obsidian_agent.core.environment import LIBRARY, model
from obsidian_agent.core.models import GraphState, Note, SearchNotes


def search_notes_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    tool_call = state["messages"][-1].tool_calls[0] # type: ignore
    keywords = tool_call["args"]["keywords"]
    k = tool_call["args"].get("k", SearchNotes.model_fields["k"].default)
    k = int(k)
    results = LIBRARY.search_notes(keywords, k)
    content = [
        Note(name=doc.metadata["path"].name, text=doc.page_content) for doc in results
    ]

    str_content = "\n---------------\n".join(
        [f"NOTENAME: {note.name}\n {note.text}" for note in content]
    )

    return {
        "messages": [
            {
                "role": "tool",
                "content": str_content,
                "tool_call_id": tool_call["id"],
            }
        ]
    }


def create_note_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    # Get the tool call from the last message
    tool_call = state["messages"][-1].tool_calls[0] # type: ignore

    note_name = tool_call["args"]["note_name"]
    note_text = tool_call["args"]["note_text"]

    try:
        LIBRARY.put_note(note_name, note_text)
        content = f"Note: {note_name} has been created."
    except FileExistsError as e:
        content = str(e)

    return {
        "messages": [
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_call["id"],
            }
        ]
    }


def read_notes_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    tool_call = state["messages"][-1].tool_calls[0] # type: ignore
    note_name = tool_call["args"]["note_name"]
    depth = tool_call["args"].get("depth", 0)

    try:
        content = LIBRARY.get_note_with_context(note_name, depth)
    except (ValueError, FileNotFoundError) as e:
        content = str(e)

    return {
        "messages": [
            {"role": "tool", "content": content, "tool_call_id": tool_call["id"]}
        ]
    }
