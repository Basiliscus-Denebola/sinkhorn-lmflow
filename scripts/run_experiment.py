import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.experiments.runner import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one solver and save all artifacts")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--problem", help="Existing problem NPZ; overrides generator")
    parser.add_argument("--solver", choices=["sinkhorn", "lbfgs", "newton", "hard_switch", "lmflow", "sinkhorn_reduced", "mirror_sr1", "sk_log_fast", "sk_scaling_fast", "lbfgs_reduced"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.problem:
        config["problem"] = {"path": args.problem}
    if args.solver:
        config["solver"]["name"] = args.solver
    print(run_experiment(config, args.output_root))


if __name__ == "__main__":
    main()
