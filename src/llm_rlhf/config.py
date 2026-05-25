"""Tiny TOML config loader.

We use `tomllib` (Python 3.11+) so there's no extra dependency. Each stage's
config TOML maps directly onto its dataclass fields, so loading is a
one-liner: `SFTConfig(**load_toml("configs/sft.toml"))`.
"""
import tomllib
from pathlib import Path
from typing import Any


def load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML file and return its top-level dict.

    If the file contains a `[section]` named after the stage (e.g. `[sft]`),
    that section's contents are returned. Otherwise the whole file is
    returned. This lets configs either be flat or grouped under a header.
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)
    # If there's a single top-level section, unwrap it for convenience.
    if len(data) == 1 and isinstance(next(iter(data.values())), dict):
        return next(iter(data.values()))
    return data
