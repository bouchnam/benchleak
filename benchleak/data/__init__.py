"""Reference-text loading for contamination scans.

A scan compares a benchmark against a *reference* set of texts the model is not
expected to have memorised. A small bundled default ships so the tool runs out of
the box; a caller can always point ``--reference`` at their own file. Domain-matched
reference data gives a cleaner signal (see ``reference.txt``).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["load_reference"]


def _parse_lines(raw: str) -> list[str]:
    """One passage per non-blank, non-comment line."""
    texts = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            texts.append(stripped)
    return texts


def load_reference(path: str | Path | None = None, limit: int | None = None) -> list[str]:
    """Load reference texts, from ``path`` if given else the bundled default.

    ``limit`` caps how many are returned (the first ``limit`` lines). The bundled
    file lives alongside this module so it travels inside the installed wheel.
    """
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
    else:
        raw = resources.files(__package__).joinpath("reference.txt").read_text(encoding="utf-8")

    texts = _parse_lines(raw)
    if not texts:
        source = str(path) if path is not None else "bundled reference.txt"
        raise ValueError(f"no reference texts found in {source}")

    return texts[:limit] if limit is not None else texts
