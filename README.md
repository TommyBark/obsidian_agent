# Obsidian Agent

Talk with your Obsidian Vault.

## Progress

- [x] PoC - retrieval and creating simple notes from vault
- [ ] Talk with your vault (with automatic retrieval of relevant information)
    - [x] Simple (Gradio?) interface
    - [ ] Advanced RAG techniques
- [ ] Quickly create new relevant short notes from URLs   

## How to Run 

1. Install dependacies from `pyproject.toml`
2. Create `.env` file - see `.env.example`
2. Run agent by launching Langgraph API with `langgraph dev` command.
3. (optional) Run Gradio Chat UI with `python src/obsidian_agent/apps/gradio_app.py` command.