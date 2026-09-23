#!/usr/bin/env python3
"""Sequential search for the muP apex (protocol registered in
speedrun/mup-apex-search.md). Ascends the prime ladder; per prime runs
d in {380, 880, 2000} x 3 seeds (split across both GPUs), then applies the
registered stop rule: two consecutive primes with mean delay <= 3 epochs at
every width. Appends rows to results/speedrun/runs.jsonl via the harness
and a search log to results/speedrun/apex_search_log.jsonl.
"""
import json
import subprocess
import sys

import numpy as np

PRIMES = [251, 281, 311, 347, 383, 421, 463]
DIMS = [380, 880, 2000]
SEEDS = [42, 43, 44]
FLOOR_EP = 3.0
RUNS = "results/speedrun/runs.jsonl"


def run_prime(p):
    procs = []
    jobs = [(d, s) for d in DIMS for s in SEEDS]
    half = len(jobs) // 2
    for gpu, chunk in ((0, jobs[:half]), (1, jobs[half:])):
        cmd = " && ".join(
            f"CUDA_VISIBLE_DEVICES={gpu} python scripts/speedrun_train.py "
            f"--p {p} --dim {d} --mup-base 236 --seed {s} "
            f"--tag apexSearch --epochs 1500" for d, s in chunk)
        procs.append(subprocess.Popen(["bash", "-c", cmd]))
    for pr in procs:
        pr.wait()


def depths(p):
    rows = [json.loads(l) for l in open(RUNS)]
    out = {}
    for d in DIMS:
        v = [r["t_gen"] - r["t_train"] for r in rows
             if r.get("tag") == "apexSearch" and r["p"] == p
             and r["dim"] == d and r["t_gen"] and r["t_train"]]
        nconv = sum(1 for r in rows if r.get("tag") == "apexSearch"
                    and r["p"] == p and r["dim"] == d and not r["t_gen"])
        out[d] = (float(np.mean(v)) if v else None, len(v), nconv)
    return out


def main():
    floor_streak = 0
    last_positive = 223  # from the registered muP pair
    for p in PRIMES:
        print(f"=== apex search: p = {p} ===", flush=True)
        run_prime(p)
        dd = depths(p)
        at_floor = all(m is not None and m <= FLOOR_EP for m, _, _ in dd.values())
        rec = {"p": p, "depths": {str(k): v for k, v in dd.items()},
               "at_floor": at_floor}
        with open("results/speedrun/apex_search_log.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if at_floor:
            floor_streak += 1
            if floor_streak >= 2:
                print(f"STOP: two consecutive floor primes; apex bound "
                      f"({last_positive}, {p}]", flush=True)
                return
        else:
            floor_streak = 0
            last_positive = p
    print(f"CEILING: grokking persists to p = {last_positive}; no floor "
          f"found within the registered ladder", flush=True)


if __name__ == "__main__":
    sys.exit(main())
