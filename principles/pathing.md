## Pathing Policy

Goal: path for best risk-adjusted improvement, not max HP or max rewards.
Evaluate entire path segments, not isolated nodes.

**Nodes:**
- **Elite:** highest-upside node. Rewards: relic + card. Cost: significant HP, requires deck readiness.
- **Monster:** standard growth node. Rewards: card reward. Cost: small-moderate HP.
- **Treasure:** free relic. Cost: none (unless forces bad route).
- **Shop:** variable. Can offer removes, cards, relics, potions. Value entirely dependent on inventory and gold.
  In **shop**: Removing a card from your deck is one of the highest-value actions in the game.
- **Rest:** offers HP recovery (rest, 30% Max HP) or card upgrade (smith). No combat risk.
- **Event:** variable outcomes, no guaranteed reward type. Common payoffs: HP, gold, cards, relics, 
  removes, max HP. Common costs: HP loss, curse, card loss, gold loss. 
  Some events are nearly free value; some are traps. Outcome depends on current deck state and choices made.
  Generally lower variance than elite; higher variance than monster.

**Aggressive pathing when:** HP healthy, deck ahead of act, potions available, recovery exists after risk.
**Defensive pathing when:** HP low, deck underperforming, poor recovery ahead, or bad boss matchup approaching.

After any fight that costs far more HP than expected: reassess and lower risk until deck improves.

### Elite readiness check
**HP gate: NEVER fight an elite below 40% HP. Rest or take a safer path first.**
Before pathing into an elite, ask:
- How can we justify the elite combat (return > HP loss so that this fight increases chance of clearing this Act)?
- Do we have backup pathing (rest? shop?) if the elite combat gives worse-than-expectation returns?