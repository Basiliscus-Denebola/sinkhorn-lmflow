import argparse
import _bootstrap  # noqa: F401
from sinkhorn_lmflow.io.config import load_config
from sinkhorn_lmflow.experiments.sr1_study import run_study


def main():
    parser = argparse.ArgumentParser(description='Run a paired single-pair positive SR1 study')
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-root',default='results')
    parser.add_argument('--resume')
    args = parser.parse_args()
    run_study(load_config(args.config),args.output_root,args.resume)


if __name__ == '__main__':
    main()
