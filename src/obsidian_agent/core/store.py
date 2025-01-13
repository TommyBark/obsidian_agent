from typing import Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# from langgraph.store.postgres import PostgresStore


def checkpoint_factory(
    checkpoint_type: str, connection_string: Optional[str] = None
) -> BaseCheckpointSaver:
    if checkpoint_type == "sqlite":
        import sqlite3

        if connection_string is None:
            raise ValueError("Checkpoint path must be provided for sqlite saver")
        conn = sqlite3.connect(connection_string)
        return SqliteSaver(conn)

    elif checkpoint_type == "memory":
        return MemorySaver()
    elif checkpoint_type == "postgres":
        raise NotImplementedError("Postgres not implemented yet")

    raise ValueError(f"Unknown checkpoint type: {checkpoint_type}")


def store_factory(
    store_type: str, connection_string: Optional[str] = None
) -> BaseStore:
    if store_type == "memory":
        return InMemoryStore()
    elif store_type == "postgres":
        raise NotImplementedError("Postgres not implemented yet")

    raise ValueError(f"Unknown store type: {store_type}")
