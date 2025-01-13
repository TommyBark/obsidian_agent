from dataclasses import dataclass
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


@dataclass
class ValueRange:
    lo: int
    hi: int


class UpdateMemory(TypedDict):
    """Decision on what memory type to update"""

    update_type: Literal["user", "instructions", "new_note"]


class ReadNote(TypedDict):
    """Decision on reading a note and recursively also its linked notes with specified depth"""

    note_name: str
    depth: Annotated[int, ValueRange(0, 2)]


class SemanticSearch(BaseModel):
    """Decision on searching notes based on keywords"""

    keywords: str = Field(description="Keywords to search notes for.")
    k: int = Field(description="Number of notes to return.", default=5)


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
