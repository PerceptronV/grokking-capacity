# Retired: speedrun harness (lenient memorisation clock)

Retired 2026-09-22. `speedrun_train.py` scored training accuracy in eval
mode (dropout off), whereas the canonical trainers (`gc-speed`, `gc-groks`)
score the running train-mode accuracy with dropout active. On identical
cells the harness clock ran 2 to 14 times faster (p = 149: 49/35/27 epochs
against the registry's 670/90/63 at d = 96/112/128). Every memorisation
time and every delay derived from `results/speedrun/archive/runs_*.jsonl`
is therefore on a different convention from the framework's laws and must
not be compared with registry values. Generalisation times (`t_gen`, val
accuracy >= 99 % in eval mode) match the registry convention and remain
usable as such.

Per-layer scaling now lives in the canonical code path (`--mup-base`,
identifying field `mup_base` in wallow.toml); the matched-protocol re-runs
are `configs/mup_depth_series.yaml`, `configs/mup_laws.yaml`,
`configs/band_gaps.yaml`. The scripts here are kept for provenance of the
superseded records in docs/.../theory/speedrun/*.md.
