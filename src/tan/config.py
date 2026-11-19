"""YAML config loading with simple nested overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
REGIONS_FILE = REPO_ROOT / "configs" / "regions.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``base`` with ``overrides`` merged in recursively."""
    out = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: str | Path = DEFAULT_CONFIG, overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    cfg = load_yaml(path)
    return deep_update(cfg, overrides) if overrides else cfg


def load_regions(path: str | Path = REGIONS_FILE) -> dict[str, dict[str, Any]]:
    return load_yaml(path)


def dotted_to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """{'a.b': 1, 'c': 2} → {'a': {'b': 1}, 'c': 2}."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        *parents, leaf = str(key).split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return out


def parse_set(items: list[str]) -> dict[str, Any]:
    """Turn ['a.b=1', 'c=x'] into {'a': {'b': 1}, 'c': 'x'} (values parsed as YAML)."""
    return dotted_to_nested({k: yaml.safe_load(v) for k, _, v in (i.partition("=") for i in items)})


def get_dotted(cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
