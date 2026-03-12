"""Load system prompts from markdown files."""

from __future__ import annotations

from pathlib import Path


class SystemPromptLoader:
    """Load system prompt files from a directory. Each .md file becomes a keyed prompt."""

    def __init__(self, prompts_dir: str | Path):
        self.dir = Path(prompts_dir)
        self._cache: dict[str, str] = {}

    def load_all(self) -> dict[str, str]:
        """Load all .md files, keyed by stem name. Strips trailing whitespace."""
        self._cache.clear()
        if self.dir.exists():
            for f in sorted(self.dir.rglob("*.md")):
                self._cache[f.stem] = f.read_text().strip()
        return self._cache

    def get(self, key: str) -> str:
        """Get a system prompt by key. Raises KeyError if not found."""
        if not self._cache:
            self.load_all()
        if key not in self._cache:
            raise KeyError(f"System prompt '{key}' not found. Available: {list(self._cache.keys())}")
        return self._cache[key]
