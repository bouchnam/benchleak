"""Reference-text loading for contamination scans.

A scan compares a benchmark against a *reference* set of texts the model is not
expected to have memorised. Bundled defaults ship so the tool runs out of the
box, selected by the benchmark's *domain* so that the comparison does not
confound domain with memorisation: math benchmarks are compared against original
math word problems, everything else against original general prose. A caller can
always point ``--reference`` at their own file instead.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

__all__ = ["load_reference", "BUNDLED_REFERENCES"]

# Bundled reference files by benchmark domain. Every passage in them was written
# for this project and never published elsewhere, so no released model can have
# trained on those exact strings.
BUNDLED_REFERENCES = {
    "general": "reference.txt",
    "math": "reference-math.txt",
}


def _parse_lines(raw: str) -> list[str]:
    """One passage per non-blank, non-comment line."""
    texts = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            texts.append(stripped)
    return texts


def load_reference(
    path: str | Path | None = None,
    limit: int | None = None,
    *,
    domain: str = "general",
) -> list[str]:
    """Load reference texts, from ``path`` if given else the bundled set for ``domain``.

    ``limit`` caps how many are returned (the first ``limit`` lines). A ``domain``
    without a bundled set falls back to the general one. The bundled files live
    alongside this module so they travel inside the installed wheel.
    """
    if path is not None:
        raw = Path(path).read_text(encoding="utf-8")
        source = str(path)
    else:
        filename = BUNDLED_REFERENCES.get(domain, BUNDLED_REFERENCES["general"])
        raw = resources.files(__package__).joinpath(filename).read_text(encoding="utf-8")
        source = f"bundled {filename}"

    texts = _parse_lines(raw)
    if not texts:
        raise ValueError(f"no reference texts found in {source}")

    return texts[:limit] if limit is not None else texts
