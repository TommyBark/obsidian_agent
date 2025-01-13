import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field
from trustcall import create_extractor

import obsidian_agent.core.configuration as configuration
from obsidian_agent.core.store import checkpoint_factory, store_factory
from obsidian_agent.utils.common import Spy, extract_tool_info
from obsidian_agent.utils.obsidian import ObsidianLibrary

OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")

if OBSIDIAN_VAULT_PATH is None:
    raise ValueError("Please set the OBSIDIAN_VAULT_PATH environment variable.")
if VECTOR_STORE_PATH is None:
    raise ValueError("Please set the VECTOR_STORE_PATH environment variable.")

LIBRARY = ObsidianLibrary(path=OBSIDIAN_VAULT_PATH, vector_store_path=VECTOR_STORE_PATH)


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@dataclass
class ValueRange:
    lo: int
    hi: int


# Update memory tool
class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""

    update_type: Literal["user", "instructions", "new_note"]


class ReadNote(TypedDict):
    """Decision on reading a note and recursively also its linked notes with specified depth"""

    note_name: str
    depth: Annotated[int, ValueRange(0, 2)]


class SemanticSearch(TypedDict):
    """Decision on searching notes based on keywords"""

    keywords: str
    k: int


model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# User profile schema
class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""

    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user, such as family members, friends, or coworkers",
        default_factory=list,
    )
    interests: list[str] = Field(
        description="Interests that the user has", default_factory=list
    )


# ToDo schema
class Note(BaseModel):
    name: str = Field(description="The task to be completed.")
    text: str = Field(description="Estimated time to complete the task (minutes).")


# Create the Trustcall extractor for updating the user profile
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

# Chatbot instruction for choosing what to update and what tools to call
MODEL_SYSTEM_MESSAGE = """{assistant_role}

Your main task is to retrieve and create new Obsidian notes from user.

Obsidian notes follow Markdown syntax but introduce a few additional features. Each note can be linked to other notes, forming a network of knowledge. 
Links are created by wrapping the note's name in double brackets, like this: [[note name]].

You have a long term memory which keeps track of three things:
1. The user's profile (general information about them) 
2. The user's personal note library
3. General instructions for creating new notes

Here is the current User Profile (may be empty if no information has been collected yet):
<user_profile>
{user_profile}
</user_profile>


Here are the current user-specified preferences for creating new notes (may be empty if no preferences have been specified yet):
<instructions>
{instructions}
</instructions>

Here are your instructions for reasoning about the user's messages:

1. Reason carefully about the user's messages as presented below. 

2. Available tools:
2a. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
2b. Decide if the new note should be created as demanded by the user
- If user asks you to create new note, create it by using UpdateMemory tool with type `new_note`
- If the user has specified preferences for how to create new notes, update the instructions by calling UpdateMemory tool with type `instructions`
2c. Decide if the user wants to read a note or search through notes
- If the user asks you to read a note, read it by calling ReadNote tool with the note name (from the user) and the depth of how many linked notes to read (usually from 0-3, default 0)
- If the user asks you to search notes, search it by calling SemanticSearch tool with the keywords and the number of notes to return (default 5)
- You currently do not have ability to update existing notes. If user asks for it inform him that you are not able to do it.

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you have created a new note
- Tell the user that you have updated instructions

4. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""

# Trustcall instruction
TRUSTCALL_INSTRUCTION = """Reflect on following interaction.

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

# Instructions for updating the ToDo list
CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to create a new notes. 

Use any feedback from the user to update how they like their new notes to be created.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""


# Node definitions
def obsidian_assistant_node(
    state: GraphState, config: RunnableConfig, store: BaseStore
):
    """Load memories from the store and use them to personalize the chatbot's response."""

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    assistant_role = configurable.assistant_role

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    if memories:
        user_profile = memories[0].value
    else:
        user_profile = None

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    if memories:
        instructions = memories[0].value
    else:
        instructions = ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        assistant_role=assistant_role,
        user_profile=user_profile,
        instructions=instructions,
    )

    # Respond using memory as well as the chat history
    response = model.bind_tools(
        [UpdateMemory, ReadNote, SemanticSearch], parallel_tool_calls=False
    ).invoke([SystemMessage(content=system_msg)] + state["messages"])

    return {"messages": [response]}


def update_profile_node(state: GraphState, config: RunnableConfig, store: BaseStore):
    """Reflect on the chat history and update the memory collection."""

    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "Profile"
    existing_memories = (
        [
            (existing_item.key, tool_name, existing_item.value)
            for existing_item in existing_items
        ]
        if existing_items
        else None
    )

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Invoke the extractor
    result = profile_extractor.invoke(
        {"messages": updated_messages, "existing": existing_memories}
    )

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            rmeta.get("json_doc_id", str(uuid.uuid4())),
            r.model_dump(mode="json"),
        )
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated profile",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def search_notes_node(state: GraphState, config: RunnableConfig, store: BaseStore):

    tool_call = state["messages"][-1].tool_calls[0]
    keywords = tool_call["args"]["keywords"]
    k = tool_call["args"]["k"]

    results = LIBRARY.search_notes(keywords, k)
    print("Searching for: ", keywords, k)
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


def create_note_node(
    state: GraphState,
    config: RunnableConfig,
    store: BaseStore,
):
    """Reflect on the chat history and create a new note in the user's library according to user's instructions."""

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(
        time=datetime.now().isoformat()
    )
    updated_messages = list(
        merge_message_runs(
            messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)]
            + state["messages"][:-1]
        )
    )

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()

    # Create the Trustcall extractor for updating the ToDo list
    todo_extractor = create_extractor(
        model, tools=[Note], tool_choice="Note", enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = todo_extractor.invoke({"messages": updated_messages})

    # Save the memories from Trustcall to the store
    new_note = result["responses"][0]
    note = new_note.model_dump(mode="json")
    try:
        LIBRARY.put_note(note["name"], note["text"])
        content = f"Note: {note['name']} has been created."
    except FileExistsError as e:
        content = e

    # Respond to the tool call made in task_mAIstro, confirming the update
    tool_calls = state["messages"][-1].tool_calls

    # Extract the changes made by Trustcall and add the the ToolMessage returned to task_mAIstro
    todo_update_msg = extract_tool_info(spy.called_tools, "Note")
    return {
        "messages": [
            {
                "role": "tool",
                "content": content,
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


def read_notes_node(
    state: GraphState,
    config: RunnableConfig,
    store: BaseStore,
):
    """Read note and it's linked notes with specified depth"""
    configurable = configuration.Configuration.from_runnable_config(config)

    # Get the tool call parameters
    tool_call = state["messages"][-1].tool_calls[0]
    print(tool_call["args"])
    note_name = tool_call["args"]["note_name"]
    depth = tool_call["args"].get("depth", 0)

    # depth = tool_call["args"]["depth"]

    # Use the library's existing function
    try:
        content = LIBRARY.get_note_with_context(note_name, depth)
    except (ValueError, FileNotFoundError) as e:
        content = e

    return {
        "messages": [
            {"role": "tool", "content": content, "tool_call_id": tool_call["id"]}
        ]
    }


def update_instructions_node(
    state: GraphState, config: RunnableConfig, store: BaseStore
):
    """Reflect on the chat history and update the memory collection."""

    # Get the user ID from the config
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id

    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")

    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    new_memory = model.invoke(
        [SystemMessage(content=system_msg)]
        + state["messages"][:-1]
        + [
            HumanMessage(
                content="Please update the instructions based on the conversation"
            )
        ]
    )

    # Overwrite the existing memory in the store
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})

    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            {
                "role": "tool",
                "content": "updated instructions",
                "tool_call_id": tool_calls[0]["id"],
            }
        ]
    }


# Conditional edge
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
    """Reflect on the memories and chat history to decide whether to update the memory collection."""
    message = state["messages"][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
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


# Create the graph + all nodes
builder = StateGraph(GraphState, config_schema=configuration.Configuration)

# Nodes
builder.add_node("obsidian_assistant", obsidian_assistant_node)
builder.add_node("create_note", create_note_node)
builder.add_node("read_notes", read_notes_node)
builder.add_node("update_profile", update_profile_node)
builder.add_node("update_instructions", update_instructions_node)
builder.add_node("search_notes", search_notes_node)

# Edges
builder.add_edge(START, "obsidian_assistant")
builder.add_conditional_edges("obsidian_assistant", route_message)
builder.add_edge("create_note", "obsidian_assistant")
builder.add_edge("read_notes", "obsidian_assistant")
builder.add_edge("update_profile", "obsidian_assistant")
builder.add_edge("update_instructions", "obsidian_assistant")
builder.add_edge("search_notes", "obsidian_assistant")

# Store for long-term (across-thread) memory
across_thread_memory = store_factory("memory")  # InMemoryStore()

# Checkpointer for short-term (within-thread) memory
within_thread_memory = checkpoint_factory("memory")

# We compile the graph with the checkpointer and store
graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)
