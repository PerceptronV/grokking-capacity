#!/usr/bin/env python3
"""Render an alternative cone aggregate into an explicitly chosen output directory."""
from pathlib import Path
import argparse,json
from cone_data import ROOT,aggregate,write_json
from analyse_cone_ten_seeds import render

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort',type=Path,default=ROOT/'results/cone_ten_seeds_h100_20261106/figure/runs.json')
    ap.add_argument('--metric',default='median',help='min, median, mean, max, quantile:<q>')
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();rows=json.loads(args.cohort.read_text())
    # This preview command cannot overwrite canonical publication directories.
    protected=[ROOT/'figures',ROOT/'docs',ROOT/'results/cone_ten_seeds_h100_20261106',args.cohort.parent]
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in protected):
        ap.error('choose a separate preview directory outside the input and publication trees')
    if any(r['mup_base']!=0 for r in rows):ap.error('this renderer expects the fixed-rate cohort')
    cells=aggregate(rows,args.metric);args.out.mkdir(parents=True,exist_ok=True)
    write_json(args.out/'cells.json',cells)
    write_json(args.out/'metric.json',{'metric':args.metric,'source':str(args.cohort),'unresolved_outcomes_retained':True})
    render(cells,args.out,metric=args.metric)

if __name__=='__main__':main()
