.PHONY: setup install init-vectorstore dev run-gradio run-streamlit run

include .env
export

setup:
	@echo "Setting up virtual environment and installing dependencies..."
	uv venv
	uv pip install -e .

# Initialize vector store from Obsidian vault
init-vectorstore: setup
	@echo "Checking if vector store exists..."
	@if [ ! -d "$(VECTOR_STORE_PATH)" ]; then \
		echo "Vector store not found at $(VECTOR_STORE_PATH). Initializing..."; \
		uv run --env-file .env -- python -c "from obsidian_agent.utils.rag import create_vector_store; \
		import os; \
		create_vector_store(os.getenv('OBSIDIAN_VAULT_PATH'), os.getenv('VECTOR_STORE_PATH'))"; \
	else \
		echo "Vector store found at $(VECTOR_STORE_PATH)"; \
	fi

# Run langgraph dev server
dev: setup
	@echo "Starting Langgraph dev server..."
	uv run --env-file .env -- langgraph dev

# Run Gradio interface
run-gradio: setup
	@echo "Starting Gradio interface..."
	uv run --env-file .env -- python -m obsidian_agent.apps.gradio_app

# Run full application (langgraph server + gradio interface)
run: setup init-vectorstore

	@echo "Starting full application..."
	@(uv run --env-file .env -- langgraph dev > /dev/null 2>&1 &) && \
	sleep 3 && \
	uv run --env-file .env -- python -m obsidian_agent.apps.gradio_app