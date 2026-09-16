from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")
    return config


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(json_safe(data), ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")


def save_config(config: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(json_safe(config), sort_keys=False), encoding="utf-8")
