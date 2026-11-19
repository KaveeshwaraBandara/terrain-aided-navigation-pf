"""Run a Monte Carlo experiment defined in configs/experiments/<name>.yaml.

Usage:
    python scripts/sweep.py --experiment e1 [--quick] [--jobs 10] [--study n_sweep]
    python scripts/sweep.py --experiment e1 --collect      # only rebuild summary.csv

Results: results/<name>[_quick]/<study>/<configuration>/run_XXXX.npz + summary.csv.
Finished runs are skipped, so an interrupted sweep resumes where it stopped.
"""

from __future__ import annotations

import os

# one maths thread per worker process: the parallelism is across Monte Carlo runs
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402

from tan.runner import collect_summary, sweep  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--experiment", required=True, help="name of configs/experiments/<name>.yaml")
    ap.add_argument("--quick", action="store_true", help="3 runs, short flight (smoke test)")
    ap.add_argument("--jobs", type=int, default=10, help="worker processes (default 10)")
    ap.add_argument("--study", help="run only this study of the experiment")
    ap.add_argument("--collect", action="store_true", help="only rebuild summary.csv")
    args = ap.parse_args()
    if args.collect:
        print(collect_summary(args.experiment, args.quick))
    else:
        sweep(args.experiment, quick=args.quick, jobs=args.jobs, only=args.study)


if __name__ == "__main__":
    main()
