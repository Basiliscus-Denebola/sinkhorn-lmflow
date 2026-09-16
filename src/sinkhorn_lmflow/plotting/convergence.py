from __future__ import annotations

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .comparisons import run_label


def save_figure(fig: plt.Figure, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(paths: list[Path], output: str | Path, x: str = "iteration") -> None:
    """Residual vs iteration/wall time, retaining and labelling every failed run."""
    if x not in ("iteration", "elapsed_sec"):
        raise ValueError("x must be iteration or elapsed_sec")
    fig, ax = plt.subplots(figsize=(10, 6))
    for path in paths:
        history = pd.read_csv(path/"history.csv")
        summary = json.loads((path/"summary.json").read_text(encoding="utf-8"))
        label = run_label(path)
        if history.empty:
            ax.plot([], [], label=label + " (no finite iterate)", linestyle="--")
            continue
        y = history["residual_l1"].to_numpy()
        ax.semilogy(history[x], np.maximum(y, np.finfo(float).tiny),
                    label=label, linestyle="-" if summary["converged"] else "--", linewidth=1.5)
        if not summary["converged"]:
            ax.scatter(history[x].iloc[-1], max(y[-1], np.finfo(float).tiny), marker="x", color="red", zorder=5)
    ax.set(xlabel="Iteration" if x == "iteration" else "Wall time (s), including diagnostics",
           ylabel="Marginal residual (max of L1 defects)", title="Saved-run convergence")
    ax.grid(True, which="both", alpha=.2)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_figure(fig, output)


def plot_theta_comparison(paths: list[Path], output: str | Path) -> None:
    """Fixed-theta comparison using the same saved convergence records."""
    plot_convergence(paths, output, x="iteration")
