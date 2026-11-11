from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from ..types import SolverState


class ThetaPolicy(Protocol):
    def __call__(self, state: SolverState) -> float:
        ...


@dataclass(frozen=True)
class ConstantTheta:
    theta: float

    def __post_init__(self) -> None:
        if not 0 <= self.theta <= 1:
            raise ValueError("theta must be in [0,1]")

    def __call__(self, state: SolverState) -> float:
        return self.theta
