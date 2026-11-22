"""
Pytest configuration for the invoice-extractor test suite.

Sets required environment variables at import time — before any src module
is imported — so that module-level `os.environ["OPENROUTER_MODEL"]` calls
in llm.py do not raise KeyError during test collection.

If a real .env file exists in the project root, python-dotenv will load it
(for integration tests that do make real API calls, e.g. smoke tests).
For unit tests that mock the API, the values set here are used instead.
"""

import os
import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set env defaults BEFORE any src module is imported.
# os.environ.setdefault keeps existing values (from .env or the shell).
os.environ.setdefault("OPENROUTER_API_KEY", "sk-test-fake-key-for-tests")
os.environ.setdefault("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl")
os.environ.setdefault("PRICE_IN_PER_MTOK", "0.06")
os.environ.setdefault("PRICE_OUT_PER_MTOK", "0.35")
os.environ.setdefault("PRICE_DATE", "2026-11-22")

