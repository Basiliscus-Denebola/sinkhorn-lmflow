import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.plotting.energy_distance import generate_report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--report', default='docs/energy_distance_v0_report.md')
    args = parser.parse_args()
    generate_report(args.run_dir, args.report, audit=args.audit)
