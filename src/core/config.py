"""Small YAML loader; no model or pipeline initialization."""

from pathlib import Path
from typing import Any, Union

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path: Union[str, Path] = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML mapping. Explicit relative paths use the working directory."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Config file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in config file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config
