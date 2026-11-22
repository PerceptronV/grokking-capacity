"""Recompute the registered paper analyses without editing manuscript sources.

Requires the local run registries, traces, and frozen law-fit artifacts. Use
--out to compare a fresh analysis with the published results in isolation.
"""
import argparse
from pathlib import Path

from .paths import RESULTS

STAGES = ("crossings", "statistics", "models", "supplement")


def run(out=RESULTS, stages=STAGES, bootstraps=1000):
    from . import crossings, models, statistics, supplement

    out = Path(out)
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown analysis stages: {sorted(unknown)}")
    if bootstraps < 1:
        raise ValueError("Bootstrap count must be positive")
    for stage in STAGES:
        if stage not in stages:
            continue
        print(f"Paper analysis: {stage}", flush=True)
        destination = out / stage
        if stage == "crossings":
            crossings.run_analysis(destination, bootstraps=bootstraps)
        elif stage == "statistics":
            statistics.run(destination)
        elif stage == "models":
            models.run(destination, statistics=out / "statistics")
        else:
            supplement.run(destination, statistics=out / "statistics")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS,
                        help="Analysis output directory (default: %(default)s)")
    parser.add_argument("--stage", choices=STAGES, action="append",
                        help="Run only selected stages, in dependency order. "
                             "Models and supplement require statistics outputs.")
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Dense-intersection bootstrap draws (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    run(args.out, args.stage or STAGES, args.bootstrap)


if __name__ == "__main__":
    main()
