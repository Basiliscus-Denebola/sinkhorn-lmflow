from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re
import uuid


def create_run_dir(root: str | Path, name: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name).strip(".") or "run"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    path = Path(root) / f"{stamp}_{safe}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path
