import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.plotting.comparisons import find_runs, group_runs
from sinkhorn_lmflow.plotting.convergence import plot_convergence
from sinkhorn_lmflow.plotting.spectral import plot_spectral


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved results only, grouped by problem and tolerance")
    parser.add_argument("path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    root = Path(args.output) if args.output else Path(args.path)/"figures"
    for group, paths in group_runs(find_runs(args.path)).items():
        plot_convergence(paths, root/group/"residual_iteration", "iteration")
        plot_convergence(paths, root/group/"residual_time", "elapsed_sec")
        plot_spectral(paths, root/group/"spectrum")
    print(root)


if __name__ == "__main__":
    main()
