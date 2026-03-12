"""Split a multi-run agent log into individual run files."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def split_runs(log_path: str, out_dir: str | None = None) -> list[str]:
    """Split a multi-run log into per-run files. Returns list of output paths."""
    log_path = Path(log_path)
    if out_dir:
        out = Path(out_dir)
    else:
        out = log_path.parent / "runs"
    out.mkdir(parents=True, exist_ok=True)

    lines = log_path.read_text().splitlines(keepends=True)

    # Find run boundaries
    starts = []
    for i, line in enumerate(lines):
        if "Auto-starting first run" in line or "Auto-starting run" in line:
            starts.append(i)

    if not starts:
        print(f"No runs found in {log_path}")
        return []

    output_files = []
    for run_idx, start_line in enumerate(starts):
        end_line = starts[run_idx + 1] if run_idx + 1 < len(starts) else len(lines)
        run_lines = lines[start_line:end_line]

        floor = "?"
        for l in run_lines:
            m = re.search(r"RUN ENDED \(Floor (\d+)\)", l)
            if m:
                floor = m.group(1)
            m2 = re.search(r"=== VICTORY", l)
            if m2:
                floor = "WIN"

        filename = out / f"run_{run_idx + 1:02d}_floor{floor}.log"
        with open(filename, "w") as f:
            f.writelines(run_lines)
        output_files.append(str(filename))
        print(f"Run {run_idx + 1}: {len(run_lines)} lines -> {filename.name}")

    print(f"\nSplit {len(starts)} runs into {out}")
    return output_files


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python split_runs.py <log_file> [output_dir]")
        sys.exit(1)
    split_runs(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
