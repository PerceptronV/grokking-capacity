#!/usr/bin/env bash
set -euo pipefail
cd /home/ubuntu/vincent-in-georgia/predicting-grokking
campaign="$PWD/results/capacity_h100_matched_20261103"
until test -f "$campaign/completion.json"; do sleep 10; done
python - "$campaign/completion.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));assert len(x)==96 and all(r['returncode']==0 for r in x), 'Training failed; publication not updated'
PY
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/update_matched_capacity_paper.py
python scripts/verify_matched_capacity_checkpoints.py
python scripts/report_matched_capacity_update.py
mkdir -p "$campaign/build"
cd 'docs/papers/2026 Grokking Capacity'
paper_source="$PWD"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$campaign/build" paper-neurips-revised.tex > "$campaign/build/pass1.log" 2>&1
# Run bibtex from the paper directory so references and local styles resolve.
(cd "$campaign/build" && BIBINPUTS="$paper_source:" BSTINPUTS="$paper_source:" bibtex paper-neurips-revised) > "$campaign/build/bibtex.log" 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$campaign/build" paper-neurips-revised.tex > "$campaign/build/pass2.log" 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$campaign/build" paper-neurips-revised.tex > "$campaign/build/pass3.log" 2>&1
cp "$campaign/build/paper-neurips-revised.pdf" paper-neurips-revised.pdf
printf 'Publication update and build finished.\n'
