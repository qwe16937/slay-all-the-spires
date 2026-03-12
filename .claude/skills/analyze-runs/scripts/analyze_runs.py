"""Analyze per-run log files for deckbuilding, pathing, and combat patterns."""
from __future__ import annotations

import re
import sys
import json
from pathlib import Path


def analyze_run(filepath: str) -> dict:
    """Analyze a single run log. Returns structured data."""
    with open(filepath) as f:
        lines = [l.rstrip() for l in f.readlines()]

    current_floor = 0
    death_floor = "?"
    victory = False
    neow_choice = None

    cards_taken = []
    cards_bought = []
    cards_removed = []
    upgrades = []
    rests = []
    skips = 0
    path_seq = []

    combats = []
    current_combat = None
    in_combat = False
    replans = 0

    for line in lines:
        # Floor/screen tracking
        m = re.match(r"Floor (\d+) \| Screen: (\w+) \| HP: (\d+)/(\d+)", line)
        if m:
            new_floor = int(m.group(1))
            screen = m.group(2)
            hp = int(m.group(3))
            max_hp = int(m.group(4))

            if screen == "combat" and not in_combat:
                in_combat = True
                current_combat = {
                    "floor": new_floor,
                    "start_hp": hp,
                    "max_hp": max_hp,
                    "end_hp": hp,
                    "turns": 0,
                    "cards_played": [],
                }

            if in_combat and screen != "combat":
                if current_combat:
                    current_combat["end_hp"] = hp
                    combats.append(current_combat)
                    current_combat = None
                in_combat = False

            if in_combat and current_combat:
                current_combat["end_hp"] = hp

            current_floor = new_floor

        # Run end
        dm = re.search(r"RUN ENDED \(Floor (\d+)\)", line)
        if dm:
            death_floor = dm.group(1)
        if "=== VICTORY" in line:
            victory = True

        # Neow choice
        if current_floor == 0 and "LLM chose:" in line and not neow_choice:
            neow_choice = line.split("LLM chose:")[-1].strip()[:120]

        # Card taken (only "Action:" lines to avoid duplicates from "LLM chose:" lines)
        if line.startswith("Action: Action(choose_card"):
            m2 = re.search(r"card_id': '([^']+)'", line)
            if m2:
                cards_taken.append({"floor": current_floor, "card": m2.group(1)})

        # Shop buy
        if line.startswith("Action: Action(shop_buy"):
            for key in ("card_id", "relic_id", "potion_id"):
                m2 = re.search(key + r"': '([^']+)'", line)
                if m2:
                    cards_bought.append({"floor": current_floor, "item": m2.group(1), "type": key.split("_")[0]})
                    break

        # Card removal
        if line.startswith("Action: Action(shop_purge") or line.startswith("Action: Action(card_purge"):
            m2 = re.search(r"card_id': '([^']+)'", line)
            cards_removed.append({"floor": current_floor, "card": m2.group(1) if m2 else "?"})

        # Upgrade (card_select after smith)
        if line.startswith("Action: Action(card_select"):
            m2 = re.search(r"card_id': '([^']+)'", line)
            if m2:
                upgrades.append({"floor": current_floor, "card": m2.group(1)})

        # Rest
        if line.startswith("Action: Action(rest"):
            rests.append(current_floor)

        # Skip card reward
        if line.startswith("Action: Action(skip_card"):
            skips += 1

        # Path choice
        if line.startswith("Action: Action(choose_path"):
            m2 = re.search(r"symbol': '(\w)'", line)
            if m2:
                path_seq.append(m2.group(1))

        # Turn counting
        if "Turn plan" in line and in_combat and current_combat:
            current_combat["turns"] += 1

        # Cards played in combat
        if in_combat and current_combat and line.startswith("Action: Action(play_card"):
            m2 = re.search(r"card_id': '([^']+)'", line)
            if m2:
                current_combat["cards_played"].append(m2.group(1))

        # Re-plans (buffered action invalid)
        if "re-planning" in line:
            replans += 1

    # Close last combat if died in it
    if current_combat:
        combats.append(current_combat)

    # Path counts
    path_counts = {}
    for p in path_seq:
        path_counts[p] = path_counts.get(p, 0) + 1

    return {
        "death_floor": death_floor,
        "victory": victory,
        "neow": neow_choice,
        "cards_taken": cards_taken,
        "cards_bought": cards_bought,
        "cards_removed": cards_removed,
        "upgrades": upgrades,
        "rests": rests,
        "skips": skips,
        "path_counts": path_counts,
        "path_seq": "".join(path_seq),
        "combats": combats,
        "replans": replans,
    }


def format_run(run_num: str, r: dict) -> str:
    """Format a single run analysis as readable text."""
    lines = []
    path_str = " ".join("{}:{}".format(k, v) for k, v in sorted(r["path_counts"].items()))
    result = "WIN" if r["victory"] else "died F{}".format(r["death_floor"])

    lines.append("=" * 70)
    lines.append("Run {} | {} | path: {} | replans: {}".format(run_num, result, path_str, r["replans"]))
    lines.append("  Neow: {}".format(r["neow"] or "N/A"))
    lines.append("")

    # Deckbuilding
    taken_str = ", ".join("F{}:{}".format(c["floor"], c["card"]) for c in r["cards_taken"])
    bought_str = ", ".join("F{}:{}".format(c["floor"], c["item"]) for c in r["cards_bought"])
    removed_str = ", ".join("F{}:{}".format(c["floor"], c["card"]) for c in r["cards_removed"])
    upgrade_str = ", ".join("F{}:{}".format(c["floor"], c["card"]) for c in r["upgrades"])

    lines.append("  DECK: taken={} bought={} removed={} upgraded={} skipped={} rested={}".format(
        len(r["cards_taken"]), len(r["cards_bought"]), len(r["cards_removed"]),
        len(r["upgrades"]), r["skips"], len(r["rests"])))
    lines.append("    Taken: {}".format(taken_str or "none"))
    if bought_str:
        lines.append("    Bought: {}".format(bought_str))
    if removed_str:
        lines.append("    Removed: {}".format(removed_str))
    else:
        lines.append("    !! ZERO REMOVALS")
    if upgrade_str:
        lines.append("    Upgrades: {}".format(upgrade_str))
    else:
        lines.append("    !! ZERO UPGRADES")
    if r["rests"]:
        lines.append("    Rests: {}".format(", ".join("F{}".format(f) for f in r["rests"])))
    lines.append("")

    # Combat
    lines.append("  COMBAT ({} fights):".format(len(r["combats"])))
    total_hp_lost = 0
    for c in r["combats"]:
        loss = c["start_hp"] - c["end_hp"]
        total_hp_lost += max(0, loss)
        flag = ""
        if loss >= 20:
            flag = " !! BIG LOSS"
        elif loss >= 15:
            flag = " !"

        played = ", ".join(c["cards_played"][:10])
        extra = "..." if len(c["cards_played"]) > 10 else ""
        lines.append("    F{}: {}->{}hp ({}T, {} cards) [{}{}]{}".format(
            c["floor"], c["start_hp"], c["end_hp"], c["turns"],
            len(c["cards_played"]), played, extra, flag))
    lines.append("    Total HP lost in combat: {}".format(total_hp_lost))
    lines.append("")

    return "\n".join(lines)


def aggregate_stats(results: list[tuple[str, dict]]) -> str:
    """Compute aggregate stats across all runs."""
    n = len(results)
    all_cards = sum(len(r["cards_taken"]) + len(r["cards_bought"]) for _, r in results)
    all_removals = sum(len(r["cards_removed"]) for _, r in results)
    all_upgrades = sum(len(r["upgrades"]) for _, r in results)
    all_skips = sum(r["skips"] for _, r in results)
    all_rests = sum(len(r["rests"]) for _, r in results)
    zero_removal = sum(1 for _, r in results if not r["cards_removed"])
    zero_upgrade = sum(1 for _, r in results if not r["upgrades"])
    zero_skip = sum(1 for _, r in results if r["skips"] == 0)
    wins = sum(1 for _, r in results if r["victory"])

    # Death floor distribution
    floor_buckets = {"1-7": 0, "8-15": 0, "16(boss)": 0, "17-33": 0, "34+": 0, "WIN": 0}
    for _, r in results:
        if r["victory"]:
            floor_buckets["WIN"] += 1
        else:
            f = int(r["death_floor"])
            if f <= 7:
                floor_buckets["1-7"] += 1
            elif f <= 15:
                floor_buckets["8-15"] += 1
            elif f == 16:
                floor_buckets["16(boss)"] += 1
            elif f <= 33:
                floor_buckets["17-33"] += 1
            else:
                floor_buckets["34+"] += 1

    # Worst fights
    big_losses = []
    for run_num, r in results:
        for c in r["combats"]:
            loss = c["start_hp"] - c["end_hp"]
            if loss >= 15:
                big_losses.append((run_num, c["floor"], loss))
    big_losses.sort(key=lambda x: -x[2])

    lines = []
    lines.append("=" * 70)
    lines.append("AGGREGATE ANALYSIS ({} runs, {} wins)".format(n, wins))
    lines.append("=" * 70)
    lines.append("")
    lines.append("DEATH FLOOR DISTRIBUTION:")
    for bucket, count in floor_buckets.items():
        bar = "#" * count
        lines.append("  {:>10s}: {:>2d}  {}".format(bucket, count, bar))
    lines.append("")
    lines.append("DECKBUILDING:")
    lines.append("  Cards added: {} ({:.1f}/run)".format(all_cards, all_cards / n))
    lines.append("  Removals:    {} ({:.1f}/run)  Zero-removal runs: {}/{}".format(
        all_removals, all_removals / n, zero_removal, n))
    lines.append("  Upgrades:    {} ({:.1f}/run)  Zero-upgrade runs: {}/{}".format(
        all_upgrades, all_upgrades / n, zero_upgrade, n))
    lines.append("  Skips:       {} ({:.1f}/run)  Zero-skip runs:    {}/{}".format(
        all_skips, all_skips / n, zero_skip, n))
    lines.append("  Rests:       {} ({:.1f}/run)".format(all_rests, all_rests / n))
    lines.append("")
    lines.append("WORST FIGHTS (HP loss >= 15):")
    for run_num, floor, loss in big_losses[:20]:
        lines.append("  Run {} F{}: -{} HP".format(run_num, floor, loss))
    lines.append("")

    return "\n".join(lines)


def main():
    import glob

    if len(sys.argv) < 2:
        print("Usage: python analyze_runs.py <runs_dir> [output_file]")
        print("  runs_dir: directory containing per-run log files (from split_runs.py)")
        sys.exit(1)

    runs_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None

    run_files = sorted(glob.glob(str(Path(runs_dir) / "run_*.log")))
    if not run_files:
        print("No run_*.log files found in {}".format(runs_dir))
        sys.exit(1)

    results = []
    output = []

    for filepath in run_files:
        name = Path(filepath).name
        run_num = name.split("_")[1]
        r = analyze_run(filepath)
        results.append((run_num, r))
        output.append(format_run(run_num, r))

    output.append(aggregate_stats(results))
    full_output = "\n".join(output)

    if out_path:
        Path(out_path).write_text(full_output + "\n")
        print("Analysis written to: {}".format(out_path))
    else:
        print(full_output)

    # Also write JSON for programmatic use
    if out_path:
        json_path = str(Path(out_path).with_suffix(".json"))
        json_data = []
        for run_num, r in results:
            json_data.append({"run": run_num, **r})
        Path(json_path).write_text(json.dumps(json_data, indent=2, ensure_ascii=False))
        print("JSON data written to: {}".format(json_path))


if __name__ == "__main__":
    main()
