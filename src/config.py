"""
Shared configuration helpers for the HIMK implementation.

- Defines the implementation root directory.
- Loads ``config.yaml`` once and caches it.
- Provides helpers for commonly used paths (results file, thesis PDF).

All paths are resolved relative to the implementation root unless an
absolute path is supplied.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml

# Directory that contains config.yaml, src/, pipelines/, etc.
ROOT: Path = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def load_config(config_path: Optional[Path] = None) -> dict[str, Any]:
    """
    Load the main YAML config.

    If ``config_path`` is not provided, uses ``ROOT / "config.yaml"``.
    The result is cached so repeated callers in the same process are cheap.
    """
    path = Path(config_path) if config_path is not None else ROOT / "config.yaml"
    with path.open() as f:
        return yaml.safe_load(f)


def default_results_path() -> Path:
    """
    Default location for the exported evaluation results JSON used by the
    frontend. This matches the existing convention in the project.
    """
    return ROOT / "data" / "results.json"


def resolve_thesis_pdf_path() -> Path:
    """
    Resolve the thesis PDF path using the same logic everywhere:

    1. ``THESIS_PDF_PATH`` environment variable, if set and exists.
    2. ``../latex/build/Thesis.pdf`` (from ROOT), if it exists.
    3. ``../Thesis.pdf`` (from ROOT), if it exists.
    4. ``../report.pdf`` (from ROOT), if it exists.
    5. As a last resort, return the value of ``THESIS_PDF_PATH`` even if
       it does not currently exist, otherwise ``ROOT / "thesis.pdf"``.
    """
    env_value = os.environ.get("THESIS_PDF_PATH")
    if env_value:
        env_path = Path(env_value)
        if env_path.exists():
            return env_path

    candidates = [
        ROOT.parent / "latex" / "build" / "Thesis.pdf",
        ROOT.parent / "Thesis.pdf",
        ROOT.parent / "report.pdf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallbacks: if env var was set but file missing, still return that path;
    # otherwise default to where Docker stage copies the built thesis.
    if env_value:
        return Path(env_value)
    return ROOT / "thesis.pdf"

