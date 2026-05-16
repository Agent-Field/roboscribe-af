"""Task adapters — pluggable per-dataset modules.

The reasoner DAG is task-agnostic. Each TaskAdapter encapsulates:
  - which HF dataset to load
  - how episodes / frames / actions are shaped
  - how to extract keyframes
  - what counts as success for that task
  - the natural-language task description for reasoner prompts

Swap PushT → Aloha → BridgeData → any LeRobot/OXE dataset by switching
the AF_TASK env var. The reasoner code never changes.
"""

from .base import TaskAdapter
from .registry import get_adapter, register_adapter

__all__ = ["TaskAdapter", "get_adapter", "register_adapter"]
