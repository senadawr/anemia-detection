"""Input/output helpers for configuration, JSON, and artifact persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file if needed."""

    path.parent.mkdir(parents=True, exist_ok=True)


def json_default(value: Any) -> Any:
    """Convert common non-JSON types into serializable Python objects."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write a JSON file with consistent formatting."""

    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=indent, default=json_default), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read a JSON file from disk."""

    return json.loads(path.read_text(encoding="utf-8"))
