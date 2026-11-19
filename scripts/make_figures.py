"""Regenerate every figure, table and macro of an experiment from its saved results.

Usage:
    python scripts/make_figures.py --experiment e1 [--quick]
    python scripts/make_figures.py --experiment smoke --as e1   # E1 analysis on smoke results
    python scripts/make_figures.py --all

Nothing is re-run: inputs are results/<exp>[_quick]/summary.csv and the per-run .npz files.
Full runs write to report/{figures,tables,generated}; --quick runs write under
results/<exp>_quick/report/ so they never overwrite report material.
"""

from __future__ import annotations

import argparse
import importlib

EXPERIMENTS = ["e1", "e2", "e3", "e4", "e5", "e6", "e7", "visuals"]


def run(exp: str, analysis: str, quick: bool) -> None:
    mod = importlib.import_module(f"tan.analysis.{analysis.split('@')[0]}")
    mod.make(exp=exp, quick=quick)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--experiment", help="experiment name (results/<name>)")
    ap.add_argument("--as", dest="analysis", help="analysis module to use (default: same name)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--all", action="store_true", help="every experiment with results")
    args = ap.parse_args()
    if args.all:
        for e in EXPERIMENTS:
            try:
                run(e, e, args.quick)
            except FileNotFoundError as exc:
                print(f"[{e}] skipped: {exc}")
    elif args.experiment:
        run(args.experiment, args.analysis or args.experiment, args.quick)
    else:
        ap.error("--experiment or --all is required")


if __name__ == "__main__":
    main()
