.PHONY: setup install init-vectorstore dev run-gradio run-streamlit run

setup:
	@echo "Setting up virtual environment and installing dependencies..."
	uv venv
	uv pip install -e .

# Initialize vector store from Obsidian vault
init-vectorstore: setup
	@echo "Initializing vector store..."
	uv run python -c "from obsidian_agent.utils.rag import create_vector_store; \
		import os; \
		create_vector_store(os.getenv('OBSIDIAN_VAULT_PATH'), os.getenv('VECTOR_STORE_PATH'))"

# Run langgraph dev server
dev: setup
	@echo "Starting Langgraph dev server..."
	uv run langgraph dev

# Run Gradio interface
run-gradio: setup
	@echo "Starting Gradio interface..."
	uv run python -m obsidian_agent.apps.gradio_app

# Run full application (langgraph server + gradio interface)
run: setup
	@echo "Starting full application..."
	@(uv run langgraph dev > /dev/null 2>&1 &) && \
	sleep 3 && \
	uv run python -m obsidian_agent.apps.gradio_app