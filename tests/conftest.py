"""Pytest configuration for grafeo-memory tests."""

import sys
from pathlib import Path

# Add tests directory to sys.path so mock_llm can be imported
sys.path.insert(0, str(Path(__file__).parent))

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager


def make_manager(outputs, db_path=None, dims=16, **config_kwargs):
    """Create a MemoryManager with mock model and specified db_path."""
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": db_path, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(model, config, embedder=embedder)
