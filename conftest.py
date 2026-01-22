"""
Pytest configuration and fixtures.

This file sets up the import paths so that test files can properly import
modules from the project.
"""

import sys
from pathlib import Path

# Add project root and module directories to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "modules" / "01_data_loaders"))
sys.path.insert(0, str(PROJECT_ROOT / "modules" / "02_variant_processing"))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
