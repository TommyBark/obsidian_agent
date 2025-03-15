import os
import pathlib
from typing import List, Optional

from langchain_community.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class ObsidianLibrary:
    def __init__(self, path: str, vector_store_path: str):
        self.path = path

        file_paths = [*pathlib.Path(path).rglob("*.md")]
        self.file_paths = [str(path) for path in file_paths]
        self.file_names = [path.name for path in file_paths]

        embedding_model = OpenAIEmbeddings()

        self.vector_store = FAISS.load_local(
            vector_store_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )

    def get_note_content(self, note_name: str, link_exists: bool = False) -> str:

        section_name = None
        if "#" in note_name:
            section_name = note_name.split("#")[1]
            note_name = note_name.split("#")[0]
        if "|" in note_name:
            note_name = note_name.split("|")[0]

        ends_with_str = str(pathlib.Path("/", f"{note_name}.md"))
        note_paths = [path for path in self.file_paths if path.endswith(ends_with_str)]
        if (len(note_paths) == 0) and (link_exists is False):
            raise FileNotFoundError(f"Note '{note_name}' not found")
        elif (len(note_paths) == 0) and (link_exists is True):
            return f"Note '{note_name}' is empty."
        elif len(note_paths) > 1:
            raise ValueError(f"Multiple notes found with name '{note_name}'")

        note_path = note_paths[0].replace("\xa0", " ")

        with open(note_path, "r", encoding="utf-8") as f:
            text = f.read()
            text = "\nNOTE NAME: " + note_name + "\n\n" + text

        if section_name is not None:
            section_text = find_and_extract_section(text, section_name)
            if section_text is None:
                print(
                    f"Section '{section_name}' not found in note '{note_name}', giving full note."
                )
                # raise ValueError(f"Section '{section}' not found in note '{note_name}'")
            else:
                text = (
                    "\nNOTE NAME: "
                    + note_name
                    + " SECTION:"
                    + section_name
                    + "\n\n"
                    + section_text
                )
        return text

    def get_note_with_context(self, note_name: str, depth: int = 2) -> str:

        note_name = note_name.rstrip(".md")
        text = self.get_note_content(note_name, link_exists=False)
        if depth == 0:
            return text

        links = self.get_note_links(text)

        if depth > 3:
            raise ValueError(
                "Depth cannot be greater than 3, use get_all_note_links instead"
            )

        all_links = links.copy()
        for _ in range(depth - 1):
            all_links = links.copy()
            for link in links:
                all_links.extend(
                    self.get_note_links(self.get_note_content(link, link_exists=True))
                )
            links = all_links.copy()

        all_links = list(set(all_links))
        for link in all_links:
            note_content = self.get_note_content(link, link_exists=True)
            text = text + f"\n\n{note_content}"
        return text

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
                note = self.get_note_content(link, link_exists=True)
                sub_links = self.get_all_note_links(
                    self.get_note_links(note), visited_links
                )
                new_links = new_links + sub_links
        links = links + new_links
        return list(dict.fromkeys(links))

    def put_note(self, note_title: str, content: str):
        path = f"{self.path}/{note_title}.md"
        if f"{note_title}.md" in self.file_names:
            raise FileExistsError(f"Note '{note_title}' already exists")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            self.file_paths.append(path)
            self.file_names.append(f"{note_title}.md")

    def search_notes(self, keywords: str, k: int = 5) -> List[Document]:
        """Search notes in the vector store based on keywords"""
        return self.vector_store.similarity_search(keywords, k)


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


if __name__ == "__main__":
    OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH")
    obsidian = ObsidianLibrary(OBSIDIAN_VAULT_PATH)
    print(obsidian.get_note_links("Motivation"))
    print(obsidian.get_note_with_context("Motivation", depth=1))
