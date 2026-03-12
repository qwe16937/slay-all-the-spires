"""Path evaluator — scores map node choices."""

from __future__ import annotations

from dataclasses import dataclass

from sts_agent.models import MapNode
from sts_agent.state.state_store import DeckProfile, RunState


@dataclass
class PathCandidate:
    """A scored path node option."""
    node: MapNode
    score: float = 0.0
    risk: str = "medium"      # "low"/"medium"/"high"
    rationale: str = ""

    def __str__(self) -> str:
        _NAMES = {"M": "Monster", "E": "Elite", "R": "Rest", "$": "Shop",
                  "?": "Unknown", "T": "Treasure", "B": "Boss"}
        name = _NAMES.get(self.node.symbol, self.node.symbol)
        return f"({self.node.x},{self.node.y}) {name} [{self.risk}] — {self.rationale}"


class PathEvaluator:
    """Scores path choices based on HP, deck strength, and run state."""

    # Base value of each node type
    _SYMBOL_BASE = {
        "M": 4.0,  # monster — moderate value
        "E": 6.0,  # elite — high value if deck can handle it
        "R": 5.0,  # rest — healing/upgrade
        "$": 5.0,  # shop — removal/purchase
        "?": 4.5,  # unknown — variable
        "T": 5.5,  # treasure — free relic
    }

    def evaluate(
        self,
        nodes: list[MapNode],
        run_state: RunState,
        deck_profile: DeckProfile,
    ) -> list[PathCandidate]:
        candidates = []
        for node in nodes:
            candidate = self._score_node(node, run_state, deck_profile)
            candidates.append(candidate)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def _score_node(
        self,
        node: MapNode,
        run_state: RunState,
        deck_profile: DeckProfile,
    ) -> PathCandidate:
        base = self._SYMBOL_BASE.get(node.symbol, 4.0)
        risk = "medium"
        rationale_parts = []
        hp_pct = run_state.hp / max(run_state.max_hp, 1) if run_state.max_hp > 0 else 0.5

        sym = node.symbol

        if sym == "E":
            # Elite scoring: high reward but risky
            if hp_pct >= 0.7 and deck_profile.frontload_score >= 4.0:
                base += 2.0
                risk = "medium"
                rationale_parts.append("good HP + deck for elite")
            elif hp_pct >= 0.5:
                base += 0.5
                risk = "medium"
                rationale_parts.append("moderate elite risk")
            else:
                base -= 3.0
                risk = "high"
                rationale_parts.append("low HP, elite is dangerous")

            # Risk posture adjustment
            if run_state.risk_posture == "aggressive":
                base += 1.0
            elif run_state.risk_posture == "defensive":
                base -= 1.5

        elif sym == "R":
            # Rest more valuable at low HP
            if hp_pct < 0.5:
                base += 2.5
                risk = "low"
                rationale_parts.append("need healing")
            elif hp_pct < 0.7:
                base += 1.0
                risk = "low"
                rationale_parts.append("moderate HP, upgrade opportunity")
            else:
                base -= 0.5
                risk = "low"
                rationale_parts.append("high HP, rest less needed")

        elif sym == "$":
            # Shop more valuable with gold and removal needs
            if run_state.gold >= 75:
                base += 1.0
                rationale_parts.append("have gold for purchases")
            if deck_profile.strike_count >= 4 or deck_profile.curse_count > 0:
                base += 1.5
                rationale_parts.append("need removal")
            risk = "low"

        elif sym == "?":
            # Unknown events — moderate risk/reward
            if hp_pct >= 0.6:
                base += 0.5
                risk = "low"
                rationale_parts.append("healthy for events")
            else:
                base -= 0.5
                risk = "medium"
                rationale_parts.append("events can cost HP")

        elif sym == "M":
            risk = "low" if hp_pct >= 0.5 else "medium"
            rationale_parts.append("standard fight")

        elif sym == "T":
            risk = "low"
            rationale_parts.append("free relic")

        return PathCandidate(
            node=node,
            score=max(0.0, min(10.0, base)),
            risk=risk,
            rationale=", ".join(rationale_parts) if rationale_parts else "standard",
        )
