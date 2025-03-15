import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Add the parent directory of the current file to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def test_initialize_environment():
    """Test that initialize_environment correctly initializes the ObsidianLibrary."""
    # Import inside the test to avoid immediate execution
    with (
        patch.dict(
            os.environ,
            {
                "OBSIDIAN_VAULT_PATH": "/path/to/vault",
                "VECTOR_STORE_PATH": "/path/to/store",
            },
        ),
        patch("obsidian_agent.utils.obsidian.ObsidianLibrary") as mock_library,
    ):

        # Mock the ObsidianLibrary constructor
        mock_library.return_value = MagicMock()

        # Import and call initialize_environment
        from obsidian_agent.core.environment import initialize_environment

        library = initialize_environment()

        # Verify ObsidianLibrary was initialized with correct paths
        mock_library.assert_called_once_with(
            path="/path/to/vault", vector_store_path="/path/to/store"
        )

        # Verify the return value
        assert library == mock_library.return_value


def test_initialize_environment_missing_vault_path():
    """Test that initialize_environment raises an error when OBSIDIAN_VAULT_PATH is missing."""
    # Import inside the test to avoid immediate execution
    with patch.dict(
        os.environ,
        {"OBSIDIAN_VAULT_PATH": "", "VECTOR_STORE_PATH": "/path/to/store"},
        clear=True,
    ):

        # Import and call initialize_environment, expect ValueError
        from obsidian_agent.core.environment import initialize_environment

        with pytest.raises(ValueError) as exc_info:
            initialize_environment()

        # Verify the error message
        assert "OBSIDIAN_VAULT_PATH" in str(exc_info.value)


def test_initialize_environment_missing_vector_store_path():
    """Test that initialize_environment raises an error when VECTOR_STORE_PATH is missing."""
    # Import inside the test to avoid immediate execution
    with patch.dict(
        os.environ,
        {"OBSIDIAN_VAULT_PATH": "/path/to/vault", "VECTOR_STORE_PATH": ""},
        clear=True,
    ):

        # Import and call initialize_environment, expect ValueError
        from obsidian_agent.core.environment import initialize_environment

        with pytest.raises(ValueError) as exc_info:
            initialize_environment()

        # Verify the error message
        assert "VECTOR_STORE_PATH" in str(exc_info.value)


def test_model_initialization_openai():
    """Test that the model is correctly initialized for OpenAI."""
    # Import inside the test to avoid immediate execution
    with (
        patch.dict(
            os.environ,
            {
                "OBSIDIAN_VAULT_PATH": "/path/to/vault",
                "VECTOR_STORE_PATH": "/path/to/store",
                "MODEL_NAME": "gpt-4o-mini",
            },
        ),
        patch("obsidian_agent.core.environment.ChatOpenAI") as mock_openai,
        patch("obsidian_agent.utils.obsidian.ObsidianLibrary"),
    ):

        # Mock the ChatOpenAI constructor
        mock_openai.return_value = MagicMock()

        # Import the module to trigger model initialization
        import obsidian_agent.core.environment

        # Verify ChatOpenAI was initialized with correct model
        mock_openai.assert_called_once_with(model="gpt-4o-mini", temperature=0)


def test_model_initialization_gemini():
    """Test that the model is correctly initialized for Gemini."""
    # Import inside the test to avoid immediate execution
    with (
        patch.dict(
            os.environ,
            {
                "OBSIDIAN_VAULT_PATH": "/path/to/vault",
                "VECTOR_STORE_PATH": "/path/to/store",
                "MODEL_NAME": "gemini-2.0-flash",
            },
        ),
        patch("obsidian_agent.core.environment.ChatGoogleGenerativeAI") as mock_gemini,
        patch("obsidian_agent.utils.obsidian.ObsidianLibrary"),
    ):

        # Mock the ChatGoogleGenerativeAI constructor
        mock_gemini.return_value = MagicMock()

        # Import the module to trigger model initialization
        import obsidian_agent.core.environment

        # Verify ChatGoogleGenerativeAI was initialized with correct model
        mock_gemini.assert_called_once_with(model="gemini-2.0-flash", temperature=0)


def test_model_initialization_unknown():
    """Test that the model initialization raises an error for unknown models."""
    # Import inside the test to avoid immediate execution
    with (
        patch.dict(
            os.environ,
            {
                "OBSIDIAN_VAULT_PATH": "/path/to/vault",
                "VECTOR_STORE_PATH": "/path/to/store",
                "MODEL_NAME": "unknown-model",
            },
        ),
        patch("obsidian_agent.utils.obsidian.ObsidianLibrary"),
    ):

        # Import the module to trigger model initialization, expect ValueError
        with pytest.raises(ValueError) as exc_info:
            import obsidian_agent.core.environment

        # Verify the error message
        assert "Unknown model name" in str(exc_info.value)
