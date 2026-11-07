#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"
campaign="$repo_dir/results/normfrozen_h100_paired_20261103"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/normfrozen_analysis.py
python scripts/update_matched_mechanisms_paper.py
mkdir -p "$campaign/build"
cd 'docs/papers/2026 Grokking Capacity'
paper_source="$PWD"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$campaign/build" paper-neurips-revised.tex > "$campaign/build/pass1.log" 2>&1
(cd "$campaign/build" && BIBINPUTS="$paper_source:" BSTINPUTS="$paper_source:" bibtex paper-neurips-revised) > "$campaign/build/bibtex.log" 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$campaign/build" paper-neurips-revised.tex > "$campaign/build/pass2.log" 2>&1
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$campaign/build" paper-neurips-revised.tex > "$campaign/build/pass3.log" 2>&1
cp "$campaign/build/paper-neurips-revised.pdf" paper-neurips-revised.pdf
