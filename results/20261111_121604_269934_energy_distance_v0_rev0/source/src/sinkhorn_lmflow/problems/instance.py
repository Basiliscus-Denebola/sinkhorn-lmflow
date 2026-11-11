from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from .validation import validate_problem


@dataclass(frozen=True)
class OTProblem:
    a: np.ndarray
    b: np.ndarray
    M: np.ndarray
    epsilon: float
    name: str = "unnamed"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in ("a", "b", "M"):
            value = np.array(getattr(self, key), dtype=np.float64, copy=True)
            value.flags.writeable = False
            object.__setattr__(self, key, value)
        validate_problem(self.a, self.b, self.M, self.epsilon)
        object.__setattr__(self, "epsilon", float(self.epsilon))
