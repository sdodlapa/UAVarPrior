"""
Pytest configuration and fixtures for UAVarPrior tests.

This module provides common test fixtures and configuration for all test modules.
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add the src directory to Python path for imports
test_dir = Path(__file__).parent
project_root = test_dir.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_config_dir():
    """Provide path to sample configuration files."""
    return project_root / "config_examples"

@pytest.fixture
def test_data_dir():
    """Provide path to test data fixtures."""
    return test_dir / "fixtures"

@pytest.fixture
def mock_model_path(temp_dir):
    """Create a mock model file for testing."""
    model_file = temp_dir / "mock_model.pt"
    model_file.touch()
    return model_file

@pytest.fixture(scope="session")
def project_root_path():
    """Provide the project root directory path."""
    return project_root

# Configure pytest to handle YAML and other test-specific settings
def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "functional: marks tests as functional/end-to-end tests"
    )

# Set test environment variables
@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "DEBUG"
    yield
    # Clean up after tests
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]
