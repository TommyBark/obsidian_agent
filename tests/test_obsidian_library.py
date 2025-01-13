import os
import sys
from pathlib import Path

import pytest

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from obsidian_utils import (  # Replace 'your_module' with the actual module name
    ObsidianLibrary,
    find_and_extract_section,
)


@pytest.fixture
def setup_obsidian_vault(tmp_path):
    """
    Sets up a temporary Obsidian vault with a specific directory structure and markdown files.
    """
    # Create directories
    notes_dir = tmp_path / "notes"
    subdir = notes_dir / "subdir"
    subdir.mkdir(parents=True)

    # Define file contents
    files = {
        notes_dir
        / "NoteA.md": """# NoteA

This is Note A. It links to and [[NoteC]] but also to the whole [[NoteB]].""",
        notes_dir
        / "NoteB.md": """# NoteB

## Section1

Content of Section 1.

## Section2

Non-existant link [[NoteE]]""",
        subdir
        / "NoteC.md": """# NoteC

This is Note C. It links to [[NoteD]].""",
        tmp_path
        / "NoteD.md": """# NoteD

This is Note D. It has a link to [[NoteB#Section1]].""",
    }

    # Create files
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")

    return ObsidianLibrary(str(tmp_path))


def test_init(setup_obsidian_vault, tmp_path):
    """
    Test the initialization of ObsidianLibrary.
    """
    obsidian = setup_obsidian_vault
    expected_files = [
        str(tmp_path / "notes" / "NoteA.md"),
        str(tmp_path / "notes" / "NoteB.md"),
        str(tmp_path / "notes" / "subdir" / "NoteC.md"),
        str(tmp_path / "NoteD.md"),
    ]
    assert set(obsidian.file_paths) == set(expected_files)


def test_get_note_content_existing_note(setup_obsidian_vault):
    """
    Test retrieving content of an existing note.
    """
    obsidian = setup_obsidian_vault
    content = obsidian.get_note_content("NoteA")
    assert "# NoteA" in content
    assert "This is Note A." in content


def test_get_note_content_nonexistent_note(setup_obsidian_vault):
    """
    Test retrieving content of a non-existent note without link_exists.
    Should raise FileNotFoundError.
    """
    obsidian = setup_obsidian_vault
    with pytest.raises(FileNotFoundError):
        obsidian.get_note_content("NonExistentNote")


def test_get_note_content_nonexistent_note_with_link(setup_obsidian_vault):
    """
    Test retrieving content of a non-existent note with link_exists=True.
    Should return a specific message.
    """
    obsidian = setup_obsidian_vault
    content = obsidian.get_note_content("NonExistentNote", link_exists=True)
    assert "Note 'NonExistentNote' is empty." in content


def test_get_note_content_duplicate_notes(setup_obsidian_vault, tmp_path):
    """
    Test retrieving content when multiple notes have the same name.
    Should raise ValueError.
    """
    # Create a duplicate NoteA.md in a different directory
    duplicate_note = tmp_path / "duplicate" / "NoteA.md"
    duplicate_note.parent.mkdir()
    duplicate_note.write_text("# NoteA\n\nDuplicate Note A.", encoding="utf-8")

    obsidian = ObsidianLibrary(str(tmp_path))
    with pytest.raises(ValueError):
        obsidian.get_note_content("NoteA")


def test_get_note_content_with_section(setup_obsidian_vault):
    """
    Test retrieving a specific section from a note.
    """
    obsidian = setup_obsidian_vault
    # Modify NoteA to include sections
    note_a_path = Path(obsidian.path) / "notes" / "NoteA.md"
    note_a_content = """# NoteA

This is Note A.

## Section1

Content of Section 1.

## Section2

Content of Section 2."""
    note_a_path.write_text(note_a_content, encoding="utf-8")

    content = obsidian.get_note_content("NoteA#Section1")
    assert "## Section1" in content
    assert "Content of Section 1." in content
    assert "Content of Section 2." not in content


def test_get_note_with_context_depth_1(setup_obsidian_vault):
    """
    Test get_note_with_context with depth=1.
    """
    obsidian = setup_obsidian_vault
    context = obsidian.get_note_with_context("NoteA", depth=1)
    print(context)
    assert "# NoteA" in context
    assert "# NoteB"
    assert "# NoteC" in context
    assert "# NoteD" not in context  # NoteD is linked via NoteC, which is depth=2


def test_get_note_with_context_depth_2(setup_obsidian_vault):
    """
    Test get_note_with_context with depth=2.
    """
    obsidian = setup_obsidian_vault
    context = obsidian.get_note_with_context("NoteA", depth=2)
    print(context)
    assert "# NoteA" in context
    assert "# NoteB" in context
    assert "# NoteC" in context
    assert "# NoteD" in context
    assert "'NoteE' is empty." in context


def test_get_note_with_context_depth_3(setup_obsidian_vault):
    """
    Test get_note_with_context with depth=2.
    """
    obsidian = setup_obsidian_vault
    context = obsidian.get_note_with_context("NoteA", depth=3)
    print(context)
    assert "# NoteA" in context
    assert "NoteB SECTION:Section1" in context
    assert "# NoteC" in context
    assert "# NoteD" in context
    assert "'NoteE' is empty." in context


def test_get_note_with_context_invalid_depth(setup_obsidian_vault):
    """
    Test get_note_with_context with invalid depth (>3).
    Should raise ValueError.
    """
    obsidian = setup_obsidian_vault
    with pytest.raises(ValueError):
        obsidian.get_note_with_context("NoteA", depth=4)


def test_get_note_links(setup_obsidian_vault):
    """
    Test extracting links from a note.
    """
    obsidian = setup_obsidian_vault
    links = obsidian.get_note_links("This links to [[NoteB]] and [[NoteC]].")
    assert "NoteB" in links
    assert "NoteC" in links
    assert len(links) == 2


def test_get_note_links_with_images(setup_obsidian_vault):
    """
    Test that image links are excluded.
    """
    obsidian = setup_obsidian_vault
    note_content = "Here is an image [[image.png]] and a link [[NoteA]]."
    links = obsidian.get_note_links(note_content)
    assert "NoteA" in links
    assert "image.png" not in links
    assert len(links) == 1


def test_get_all_note_links(setup_obsidian_vault):
    """
    Test retrieving all linked notes recursively.
    """
    obsidian = setup_obsidian_vault
    all_links = obsidian.get_all_note_links(["NoteA"])
    expected_links = {"NoteA", "NoteB", "NoteC", "NoteD", "NoteE", "NoteB#Section1"}
    assert set(all_links) == expected_links


def test_put_note_success(setup_obsidian_vault):
    """
    Test adding a new note successfully.
    """
    obsidian = setup_obsidian_vault
    obsidian.put_note("NoteE", "# NoteE\n\nThis is Note E.")
    # Verify the note was created
    note_e_path = Path(obsidian.path) / "NoteE.md"
    assert note_e_path.exists()
    assert note_e_path.read_text(encoding="utf-8") == "# NoteE\n\nThis is Note E."


def test_put_note_already_exists(setup_obsidian_vault):
    """
    Test adding a note that already exists.
    Should raise FileExistsError.
    """
    obsidian = setup_obsidian_vault
    with pytest.raises(FileExistsError):
        obsidian.put_note("NoteA", "# NoteA\n\nDuplicate content.")


def test_find_and_extract_section():
    """
    Test the helper function to extract a section from text.
    """
    text = """# NoteA

Introduction.

## Section1

Content of Section 1.

## Section2

Content of Section 2.

### Subsection2.1

Content of Subsection 2.1.

# NoteB

Another note."""

    # Extract Section1
    section1 = find_and_extract_section(text, "Section1")
    assert "## Section1" in section1
    assert "Content of Section 1." in section1
    assert "## Section2" not in section1

    # Extract Section2
    section2 = find_and_extract_section(text, "Section2")
    assert "## Section2" in section2
    assert "Content of Section 2." in section2
    assert (
        "### Subsection2.1" in section2
    )  # Should include up to next header of same or higher level

    # Extract Subsection2.1
    subsection = find_and_extract_section(text, "Subsection2.1")
    assert "### Subsection2.1" in subsection
    assert "Content of Subsection 2.1." in subsection

    # Attempt to extract non-existent section
    non_existent = find_and_extract_section(text, "NonExistent")
    assert non_existent is None


def test_get_note_content_with_section_not_found(setup_obsidian_vault, capsys):
    """
    Test retrieving a section that does not exist.
    Should return the full note and print a message.
    """
    obsidian = setup_obsidian_vault
    # Modify NoteA to include sections
    note_a_path = Path(obsidian.path) / "notes" / "NoteA.md"
    note_a_content = """# NoteA

This is Note A.

## Section1

Content of Section 1."""
    note_a_path.write_text(note_a_content, encoding="utf-8")

    content = obsidian.get_note_content("NoteA#Section2")
    captured = capsys.readouterr()
    assert "Section 'Section2' not found in note 'NoteA'" in captured.out
    assert "# NoteA" in content
    assert "This is Note A." in content
    assert "## Section1" in content
    assert "Content of Section 1." in content
    assert "Section2" not in content


def test_links_to_nonexistent_notes_direct_ask(setup_obsidian_vault):
    """
    Test that links to non-existent notes are handled correctly.
    """
    obsidian = setup_obsidian_vault

    with pytest.raises(FileNotFoundError) as excinfo:
        obsidian.get_note_content("NoteE", link_exists=False)
    assert str(excinfo.value) == "Note 'NoteE' not found"


def test_links_to_nonexistent_notes_linked(setup_obsidian_vault):
    """
    Test that links to non-existent notes are handled correctly.
    """
    obsidian = setup_obsidian_vault

    context = obsidian.get_note_content("NoteE", link_exists=True)
    assert "Note 'NoteE' is empty." in context  # Since link_exists=True for NoteE


def test_notes_in_different_directories(setup_obsidian_vault):
    """
    Test that notes in different directories are correctly handled.
    """
    obsidian = setup_obsidian_vault
    # Retrieve NoteC which is in a subdirectory
    content = obsidian.get_note_content("NoteC")
    assert "# NoteC" in content
    assert "This is Note C." in content


def test_get_note_content_with_section_link(setup_obsidian_vault):
    """
    Test retrieving a note that links to a specific section within another note.
    """
    obsidian = setup_obsidian_vault
    # NoteA links to NoteB#Section1
    context = obsidian.get_note_with_context("NoteD", depth=1)
    print(context)
    # Verify that only Section1 of NoteB is included
    assert "NoteB SECTION:Section1" in context
    assert "Content of Section 1." in context
    assert "## Section2" not in context  # Section2 should not be included
