import argparse
from pathlib import Path
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.plotting.comparisons import summarize_runs


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all saved runs, including failures")
    parser.add_argument("path")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    frame = summarize_runs(args.path)
    output = Path(args.output) if args.output else Path(args.path)/"comparison.csv"
    frame.to_csv(output, index=False)
    print(frame[["problem", "solver_name", "theta", "converged", "n_iter", "elapsed_sec",
                 "residual_l1", "stop_reason"]].to_string(index=False))
    print(output)


if __name__ == "__main__":
    main()
