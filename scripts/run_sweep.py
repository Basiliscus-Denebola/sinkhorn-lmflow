import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.experiments.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an explicitly bounded experiment sweep")
    parser.add_argument("--config", default="configs/theta_sweep.yaml")
    parser.add_argument("--output-root", default="results")
    args = parser.parse_args()
    print(run_sweep(load_config(args.config), args.output_root))


if __name__ == "__main__":
    main()
