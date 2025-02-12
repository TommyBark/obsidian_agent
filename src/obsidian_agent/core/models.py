from dataclasses import dataclass
from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@dataclass
class ValueRange:
    lo: int
    hi: int


class SearchNotes(BaseModel):
    """Search notes in the vector store based on keywords."""
    keywords: str = Field(description="The keywords to search for")
    k: int = Field(default=5, description="The number of results to return")

class CreateNote(BaseModel):
    """Creates a note in the library."""
    note_name: str = Field(description="The name of the note to be created")
    note_text: str = Field(description="The content of the note to be created")

class ReadNote(BaseModel):
    """Read a note and its linked notes."""
    note_name: str = Field(description="The name of the note to read")
    depth: int = Field(default=0, description="The depth of linked notes to read")

class UpdateMemory(BaseModel):
    """Update either user profile or instructions."""
    update_type: Literal["user", "instructions"] = Field(
        description="Type of update - user for profile, instructions for custom instructions"
    )


class Profile(BaseModel):
    """This is the profile of the user you are chatting with"""

    name: Optional[str] = Field(description="The user's name", default=None)
    location: Optional[str] = Field(description="The user's location", default=None)
    job: Optional[str] = Field(description="The user's job", default=None)
    connections: list[str] = Field(
        description="Personal connection of the user",
        default_factory=list,
    )
    interests: list[str] = Field(
        description="Interests that the user has", default_factory=list
    )


class Note(BaseModel):
    name: str = Field(description="Note name.")
    text: str = Field(description="Note content.")
