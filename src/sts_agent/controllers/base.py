"""Base protocol and context for screen controllers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from sts_agent.models import GameState, Action
from sts_agent.state.state_store import StateStore
from sts_agent.card_db import CardDB
from sts_agent.monster_db import MonsterDB
from sts_agent.relic_db import RelicDB
from sts_agent.llm_client import LLMClient


@dataclass
class ControllerContext:
    """Shared context passed to all controllers."""
    state_store: StateStore
    card_db: CardDB
    monster_db: MonsterDB
    llm: LLMClient
    system_prompt: str
    messages: list[dict] = field(default_factory=list)
    relic_db: Optional[RelicDB] = None


@runtime_checkable
class ScreenController(Protocol):
    """Protocol for per-screen decision logic.

    Returns an Action, or None to signal that the controller
    could not decide (triggering fallback / legacy path).
    """

    def decide(
        self,
        state: GameState,
        actions: list[Action],
        ctx: ControllerContext,
    ) -> Optional[Action]:
        ...
