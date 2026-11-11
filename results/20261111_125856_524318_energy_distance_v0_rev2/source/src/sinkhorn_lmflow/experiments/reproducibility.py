from __future__ import annotations

from datetime import datetime, timezone
import io
from contextlib import redirect_stdout
import os
import platform
from pathlib import Path
import subprocess
import sys
import numpy as np
import scipy


def runtime_metadata(seed: int | None) -> dict:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return {"python": sys.version, "python_executable": sys.executable,
            "numpy": np.__version__, "scipy": scipy.__version__, "os": platform.platform(),
            "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
            "git_commit": commit, "git_dirty": dirty, "seed": seed,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "blas_configuration": buffer.getvalue(),
            "thread_environment": {k: os.environ.get(k) for k in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}
