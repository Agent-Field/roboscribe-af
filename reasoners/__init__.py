"""Reasoner routers — grouped by cognitive role."""

from .action import action_router
from .composer import composer_router
from .corpus import corpus_router
from .dl import dl_router
from .smoke import smoke_router
from .temporal import temporal_router
from .visual import visual_router

__all__ = [
    "action_router",
    "composer_router",
    "corpus_router",
    "dl_router",
    "smoke_router",
    "temporal_router",
    "visual_router",
]
