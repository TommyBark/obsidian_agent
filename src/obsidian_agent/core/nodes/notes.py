# obsidian_agent/core/nodes/notes.py
from datetime import datetime

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from trustcall import create_extractor

from obsidian_agent.core.environment import LIBRARY, model
from obsidian_agent.core.models import GraphState, Note, SemanticSearch
from obsidian_agent.utils.common import Spy, extract_tool_info


def search_notes_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    tool_call = state["messages"][-1].tool_calls[0]
    keywords = tool_call["args"]["keywords"]
    k = tool_call["args"].get("k", SemanticSearch.model_fields["k"].default)

    results = LIBRARY.search_notes(keywords, k)
    content = [
        Note(name=doc.metadata["path"].name, text=doc.page_content) for doc in results
    ]

    str_content = "\n---------------".join(
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
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    spy = Spy()
    todo_extractor = create_extractor(
        model, tools=[Note], tool_choice="Note", enable_inserts=True
    ).with_listeners(on_end=spy)

    result = todo_extractor.invoke({"messages": updated_messages})
    new_note = result["responses"][0]
    note = new_note.model_dump(mode="json")

    try:
        LIBRARY.put_note(note["name"], note["text"])
        content = f"Note: {note['name']} has been created."
    except FileExistsError as e:
        content = str(e)

    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def read_notes_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    tool_call = state["messages"][-1].tool_calls[0]
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
