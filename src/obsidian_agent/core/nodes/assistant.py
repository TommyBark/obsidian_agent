from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

import obsidian_agent.core.configuration as configuration
from obsidian_agent.core.environment import model
from obsidian_agent.core.models import (
    CreateNote,
    GraphState,
    ReadNote,
    SearchNotes,
    UpdateMemory,
)

MODEL_SYSTEM_MESSAGE = """{assistant_role}

Your main task is to help the user with whatever he needs possibly using notes from their Obsidian Library.

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

1. You are a helpful user's assistant. You have access to user's notes, which are fully in their ownership, therefore they can ask you to utilize them in whatever form they want. For example to create a report for them using multiple notes etc.. Reason carefully about the user's messages as presented below. 

2. Available tools:
2a. Decide whether any of the your long-term memory should be updated:
- If personal information was provided about the user, update the user's profile by calling UpdateMemory tool with type `user`
2b. Decide if the new note should be created as demanded by the user
- If user asks you to create new note, create it by using CreateNote tool with type `new_note`
- If the user has specified preferences for how to create new notes, update the instructions by calling UpdateMemory tool with type `instructions`
2c. Decide if the user wants to read a note or search through notes
- If the user asks you to read a note, read it by calling ReadNote tool with the note name (from the user) and the depth of how many linked notes to read (usually from 0-3, default 0)
- If the user asks you to search notes, search it by calling SearchNotes tool with the keywords and the number of notes to return (default 5)
- You currently do not have ability to update existing notes. If user asks for it inform him that you are not able to do it.

IMPORTANT: Call only one tool at a time. Wait for the tool's response before making another tool call.

3. Tell the user that you have updated your memory, if appropriate:
- Do not tell the user you have updated the user's profile
- Tell the user them when you have created a new note
- Tell the user that you have updated instructions

4. Respond naturally to user user after a tool call was made to save memories, or if no tool call was made."""


def obsidian_assistant_node(
    state: GraphState, config: RunnableConfig, store: BaseStore
):
    """Load memories from the store and use them to personalize the chatbot's response."""
    configurable = configuration.Configuration.from_runnable_config(config)
    user_id = configurable.user_id
    assistant_role = configurable.assistant_role

    # Retrieve profile memory
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    # Retrieve custom instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    instructions = memories[0].value if memories else ""

    system_msg = MODEL_SYSTEM_MESSAGE.format(
        assistant_role=assistant_role,
        user_profile=user_profile,
        instructions=instructions,
    )

    tools = [UpdateMemory, CreateNote, ReadNote, SearchNotes]
    response = model.bind_tools(tools=tools).invoke([SystemMessage(content=system_msg)] + state["messages"], config=config)

    return {"messages": [response]}
