import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.io.problem_io import save_problem
from sinkhorn_lmflow.experiments.runner import make_problem


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and save one reusable OT problem")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--output", default="data/problem.npz")
    args = parser.parse_args()
    print(save_problem(make_problem(load_config(args.config)["problem"]), args.output))


if __name__ == "__main__":
    main()
