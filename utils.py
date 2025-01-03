import os
from typing import List, Optional


class ObsidianLibrary:
    def __init__(self, path: str):
        self.path = path

    def get_note_content(self, note: str, section: Optional[str] = None) -> str:

        if os.path.exists(f"{self.path}/{note}.md") is False:
            raise FileNotFoundError(f"Note '{note}' not found")
        with open(f"{self.path}/{note}.md", "r", encoding="utf-8") as f:
            text = f.read()

        if section is not None:
            text = find_and_extract_section(text, section)
            if text is None:
                raise ValueError(f"Section '{section}' not found in note '{note}'")
        return text

    def get_note_with_context(self, note: str, depth: int = 2) -> str:
        pass

    def get_note_links(self, note: str) -> List[str]:
        results = []
        start = 0
        while True:
            start = note.find("[[", start)
            if start == -1:
                break
            end = note.find("]]", start)
            if end == -1:
                break
            results.append(note[start + 2 : end])
            start = end + 2
        # Remove images
        results = [
            link for link in results if ".png" not in link and ".jpg" not in link
        ]

        return results

    def get_all_note_links(
        self, links: list, visited_links: Optional[set] = None
    ) -> List[str]:

        if visited_links is None:
            visited_links = set()

        if set(links).issubset(visited_links):
            return links

        new_links = []
        for link in links:
            if link in visited_links:
                continue
            else:
                visited_links.add(link)
                note = self.get_note_content(link)
                sub_links = self.get_all_note_links(
                    self.get_note_links(note), visited_links
                )
                new_links = new_links + sub_links
        links = links + new_links
        return list(dict.fromkeys(links))

    def put_note(self, note_title: str, content: str):
        path = f"{self.path}/{note_title}.md"
        if os.path.exists(path):
            raise FileExistsError(f"Note '{note_title}' already exists")
        with open(path, "w") as f:
            f.write(content)


def find_and_extract_section(text: str, search_string: str) -> Optional[str]:
    # First find the header line containing our search string
    try:
        idx = text.index(search_string)
        start = text.rfind("\n", 0, idx) + 1
        end = text.find("\n", idx)
        if end == -1:
            header_line = text[start:]
        else:
            header_line = text[start:end]

        # Verify it's a header line
        if not header_line.startswith("#"):
            return None

        # Count header level
        header_level = len(header_line) - len(header_line.lstrip("#"))

        # Find section content
        content_start = end + 1 if end != -1 else len(text)

        # Find end position
        curr_pos = content_start
        while True:
            # Find next header
            next_hash = text.find("\n#", curr_pos)
            if next_hash == -1:
                # No more headers, take all remaining text
                return header_line + "\n\n" + text[content_start:].strip()

            # Check header level
            end_hash = text.find("\n", next_hash + 1)
            if end_hash == -1:
                end_hash = len(text)

            header_text = text[next_hash + 1 : end_hash]
            curr_level = len(header_text) - len(header_text.lstrip("#"))

            if curr_level <= header_level:
                # Found header of same or higher level
                return header_line + "\n\n" + text[content_start:next_hash].strip()

            curr_pos = end_hash

    except ValueError:
        return None


# For inspecting tool calls
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )


def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories.

    Args:
        tool_calls: List of tool calls from the model
        schema_name: Name of the schema tool (e.g., "Memory", "ToDo", "Profile")
    """

    # Initialize list of changes
    changes = []

    for call_group in tool_calls:
        for call in call_group:
            if call["name"] == "PatchDoc":
                changes.append(
                    {
                        "type": "update",
                        "doc_id": call["args"]["json_doc_id"],
                        "planned_edits": call["args"]["planned_edits"],
                        "value": call["args"]["patches"][0]["value"],
                    }
                )
            elif call["name"] == schema_name:
                changes.append({"type": "new", "value": call["args"]})

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change["type"] == "update":
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Added content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n" f"Content: {change['value']}"
            )

    return "\n\n".join(result_parts)
