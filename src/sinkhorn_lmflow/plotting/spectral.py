from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .comparisons import run_label
from .convergence import save_figure


def plot_spectral(paths: list[Path], output: str | Path) -> None:
    """Saved sigma_nontrivial and 1-theta*sigma_nontrivial diagnostics."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for path in paths:
        history = pd.read_csv(path/"history.csv")
        label = run_label(path)
        for ax, column in zip(axes, ("sigma_nontrivial", "spectral_gap")):
            finite = history.dropna(subset=[column])
            ax.plot(finite["iteration"], finite[column], label=label, linewidth=1.3)
            if finite.empty:
                ax.text(.02, .05, "Some runs have no valid spectral samples; see logs",
                        transform=ax.transAxes, fontsize=7)
    axes[0].set(ylabel="Largest nontrivial singular value", title="Standardized transport spectrum")
    axes[1].set(xlabel="Iteration", ylabel="1 - theta * sigma_nontrivial")
    for ax in axes:
        ax.grid(True, alpha=.2)
    axes[0].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    save_figure(fig, output)
