"""Abstract base class for game interfaces."""

from abc import ABC, abstractmethod

from sts_agent.models import GameState, Action


class GameInterface(ABC):
    """Abstract interface — backend can be CommunicationMod, headless sim, etc."""

    @abstractmethod
    def observe(self) -> GameState:
        """Return current game state."""
        ...

    @abstractmethod
    def available_actions(self, state: GameState) -> list[Action]:
        """Return all legal actions for the given state."""
        ...

    @abstractmethod
    def act(self, action: Action) -> GameState:
        """Execute action, return new state."""
        ...

    @property
    @abstractmethod
    def is_terminal(self) -> bool:
        """True if the run has ended (win or loss)."""
        ...
