"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is listed, but keep CLI resilient.
    yaml = None  # type: ignore[assignment]


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration.

    Args:
        path: Optional custom YAML path.

    Returns:
        Parsed configuration dictionary. Missing files produce an empty config.
    """

    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists() or yaml is None:
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}
