"""Adapter registry — name → adapter instance.

Reasoners look up adapters by name (typically read from the AF_TASK env var
inside the entry reasoner). New adapters register themselves via
`register_adapter(name, builder)` at import time.
"""

import os
from typing import Callable

from .base import TaskAdapter

_REGISTRY: dict[str, Callable[[], TaskAdapter]] = {}


def register_adapter(name: str, builder: Callable[[], TaskAdapter]) -> None:
    """Register a deferred adapter builder. Builders are zero-arg callables
    that return a TaskAdapter instance — letting us defer heavy imports
    (datasets, etc.) until the adapter is actually used.
    """
    _REGISTRY[name] = builder


def get_adapter(name: str | None = None) -> TaskAdapter:
    """Resolve an adapter by name, defaulting to AF_TASK env var → "pusht"."""
    resolved = name or os.getenv("AF_TASK", "pusht")
    if resolved not in _REGISTRY:
        raise KeyError(
            f"No adapter registered for task '{resolved}'. "
            f"Registered: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[resolved]()
