# Obsidian Agent

<p align="center">
  <strong>An AI assistant that helps you interact with your Obsidian vault through natural conversation.</strong>
</p>

<p align="center">
  <a href="#usage">Usage</a> •
  <a href="#examples">Examples</a> •
  <a href="#development">Development</a> •
  <a href="#license">License</a>
</p>

## Overview

Obsidian Agent is an AI-powered assistant that connects to your Obsidian knowledge base, allowing you to:
- Have natural conversations about your notes
- Search your vault with semantic understanding
- Create new notes from conversations or external URLs
- Navigate complex note relationships

Built with LangGraph, LangChain, and modern LLMs (currently supports OpenAI and Google models).

## Future Enhancements
Items that still need to be implemented in the near future:

### Functionality
- [ ] **Human-in-the-loop Controls**
- [ ] **Note Editing**: Implement the ability to safely update existing notes
- [ ] **Advanced RAG Techniques**: Implement hybrid search (keyword + semantic)
- [ ] **Memory Management**: Implement better conversation history and context handling
- [ ] **Voice Interface**

### Technical
- [ ] **Documentation**
- [ ] **Testing Framework**
- [ ] **Replace uv requirement**: Use standard Python package management tools instead
- [ ] **Remove Langchain API key dependency**: Not sure currently if this can be done (Thanks Langchain!)

## Usage

### Prerequisites

- Python 3.11 or higher
- uv - https://github.com/astral-sh/uv (requirement will be removed in the future)
- `.env` file (use `.env.example` template)
- An Obsidian vault
- API keys for either OpenAI or Google Gemini
- currently still requires Langchain API key (will change in the future)
- (optional) Jina.ai API key for faster and more reliable URL retrieving

### Usage

Easiest way to run is just to run 
```bash
make run
```

Alternatively to have a better control over langgraph backend and gradio frontend, you can run separetely

```bash
langgraph dev
```
This will launch the backend server on http://127.0.0.1:2024, and

```bash
python src/obsidian_agent/apps/gradio_app.py
```

## Examples

Here are some examples of what you can ask the Obsidian Agent:

### Search Your Vault

```
User: What do my notes say about meditation?
```

### Read a Specific Note

```
User: Can you read my note on Buddhism for me?
```

### Create a New Note

```
User: Create a new note called "Project Ideas" with a list of my recent project ideas we've discussed.
```

### Generate Note from URL

```
User: Can you summarize this article and create a note from it? https://example.com/interesting-article
```

### Exploring Linked Notes

```
User: What are all the notes connected to my "Personal Knowledge Management" note?
```

## Development

### Project Structure

```
obsidian-agent/
├── src/
│   └── obsidian_agent/
│       ├── apps/                  # UI applications (Gradio, Streamlit)
│       ├── core/                  # Core logic and LangGraph components
│       │   ├── nodes/             # Graph nodes for different functionalities
│       │   └── models.py          # Data models
│       └── utils/                 # Utility functions
├── tests/                         # Test files
└── langgraph.json                 # LangGraph configuration
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
