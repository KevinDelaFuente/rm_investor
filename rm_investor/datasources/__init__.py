"""Data-source adapters and the registry that wires them to capabilities."""
from .registry import DataHub, get_hub

__all__ = ["DataHub", "get_hub"]
