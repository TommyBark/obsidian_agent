"""Obsidian Agent - An LLM agent assistant for Obsidian library."""

from pathlib import Path

import tomli


def get_version():
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        return tomli.load(f)["project"]["version"]


__version__ = get_version()
