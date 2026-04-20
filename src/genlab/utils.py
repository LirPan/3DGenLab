from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if loaded is not None else {}


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return the Path."""
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def ensure_parent_dir(file_path: str | Path) -> Path:
    """Create parent directory for a file path and return the file Path."""
    file_obj = Path(file_path)
    file_obj.parent.mkdir(parents=True, exist_ok=True)
    return file_obj


def log_step(message: str) -> None:
    """Print a consistently formatted step log line."""
    print(f"[GenLab] {message}")


def get_stem(path: str | Path) -> str:
    """Return filename stem from path-like input."""
    return Path(path).stem
