#!/usr/bin/env python3
"""Re-render the paper's figures with below-plot explanations and PDF/PNG pairs.

Uses saved inputs, checks sweep data against the existing vector plots, and
never updates training results, fitted-law artifacts or paper wording.
"""
from pathlib import Path
from collections import Counter
import importlib.util
import json
import shutil
import subprocess
import sys
import numpy as np
import pandas as pd
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figure_explanations_below_20261106'
PAPER = ROOT / 'docs/papers/2026 Grokking Capacity'
MEDIA = PAPER / 'media'
BASE = OUT / 'before'


def source_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if name.endswith('._baseline_stats'):
        text = path.read_text().replace('    ax.set_yscale("log")\n',
            '    ax.set_yscale("log")\n'
            '    ax.xaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())\n'
            '    ax.yaxis.set_minor_formatter(plt.matplotlib.ticker.NullFormatter())\n', 1)
        exec(compile(text, str(path), 'exec'), module.__dict__)
    else:
        spec.loader.exec_module(module)
    return module


def compare_scatter(old_path, generated_path, count):
    """Compare values via normalized vector marker centres, allowing font layout changes."""
    pages = [pymupdf.open(path)[0] for path in [old_path, generated_path]]
    # Some archived PDFs encode the Unicode minus tick glyph as an unextractable path.
    # Tick positions and signed observations are checked independently below.
    texts = [page.get_text().replace('−', '') for page in pages]
    assert Counter(texts[0].split()) == Counter(texts[1].split()), old_path
    points = []
    for page in pages:
        drawings = page.get_drawings()
        bounds = drawings[1]['rect']
        markers = [d for d in drawings if len(d['items']) == 8
                   and all(item[0] == 'c' for item in d['items']) and d['fill'] is not None][:count]
        assert len(markers) == count, (old_path, len(markers), count)
        points.append(np.array([[(d['rect'].x0+d['rect'].x1)/2-bounds.x0,
                                 (d['rect'].y0+d['rect'].y1)/2-bounds.y0] for d in markers])
                      / [bounds.width,bounds.height])
    maximum = float(np.max(np.abs(points[0]-points[1])))
    assert maximum < 1e-6, (old_path, maximum)
    return maximum


def main():
    from grokking_capacity.analysis.config_view import ConfigView
    from grokking_capacity.analysis import stats
    baseline = source_module(BASE/'src/grokking_capacity/analysis/stats.py',
                             'grokking_capacity.analysis._baseline_stats')
    checks = []
    for suite, axis in [('init_scale_sweep','init_scale'),('weight_decay_sweep','weight_decay'),
                        ('lr_sweep','lr'),('depth_scaling','depth'),
                        ('task_add','operation'),('task_mul','operation')]:
        folder = OUT/'inputs'/suite
        folder.mkdir(parents=True,exist_ok=True)
        candidates = [BASE/'figures'/suite/'intersection/predictiveness.csv',
                      BASE/'results/task_mul_sweep/figures/intersection/predictiveness.csv'] if suite == 'task_mul' else [BASE/'figures'/suite/'intersection/predictiveness.csv']
        frozen = next((path for path in candidates if path.exists()), None)
        if frozen:
            shutil.copy2(frozen, folder/'predictiveness.csv')
            df = pd.read_csv(frozen)
        elif (folder/'predictiveness.csv').exists():
            df = pd.read_csv(folder/'predictiveness.csv')
        else:
            view = ConfigView.from_yaml(ROOT/'configs'/f'{suite}.yaml', db_path=str(ROOT/'runs.db'))
            figure = next(x for x in view.intersection_figures if x.name == 'intersection')
            df = stats.compute_predictiveness(view, figure)
            df.to_csv(folder/'predictiveness.csv',index=False)
        for name, is_error in [('predicted_vs_empirical',False),('error_vs_'+axis,True)]:
            before = BASE/MEDIA.relative_to(ROOT)/suite/'intersection'/(name+'.pdf')
            check = folder/(name+'_baseline.pdf')
            if is_error:
                baseline.plot_error_vs_axis(df,axis,check)
                count = len(df.dropna(subset=['log_ratio',axis]))
            else:
                baseline.plot_predicted_vs_empirical(df,check)
                valid=df.dropna(subset=['predicted_onset_x','empirical_onset_x'])
                count=int(((valid['predicted_onset_x']>0)&(valid['empirical_onset_x']>0)).sum())
            delta=compare_scatter(before,check,count)
            target=MEDIA/suite/'intersection'/(name+'.pdf')
            if is_error:stats.plot_error_vs_axis(df,axis,target)
            else:stats.plot_predicted_vs_empirical(df,target)
            checks.append({'figure':str(target.relative_to(ROOT)),'points':count,
                           'max_normalized_marker_difference':delta,'displayed_values_unchanged':True})
            for ext in ['.pdf','.png']:
                dst=ROOT/'figures'/suite/'intersection'/target.with_suffix(ext).name
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target.with_suffix(ext),dst)
            print('Updated',suite,name,flush=True)
    from relocate_archived_alpha_explanations import render as relocate_alpha
    checks.extend(relocate_alpha())
    for path in (MEDIA/'alpha_sweep/intersection').glob('*'):
        if path.suffix in ['.pdf','.png']:
            dest=ROOT/'figures/alpha_sweep/intersection'/path.name
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    capacity=source_module(ROOT/'scripts/update_matched_capacity_paper.py','_capacity_figure_update')
    cap=ROOT/'results/capacity_h100_matched_20261103'
    capacity.plots(json.loads((cap/'summary.json').read_text()),json.loads((cap/'cells.json').read_text()),write_manifest=False)
    for mode in ['fixed','mup']:
        subprocess.run([sys.executable,str(ROOT/'scripts/crossing_analysis.py'),mode,'--plots-only'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/plot_cone_seedmin.py'),'--fixed-only','--final','--publish'],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    campaign=ROOT/'results/cone_ten_seeds_h100_20261106'
    for suffix in ['.pdf','.png','_plot.json']:
        name='cone_seedmin_fixed'+suffix
        shutil.copy2(campaign/'figure'/name,campaign/'live'/name)
    # These already place their legends below, or use only direct curve labels.
    for name in ['delay_with_speed_p97','delay_with_speed_p113','delay_with_speed_p139','grokking_separate_p127']:
        pdf=MEDIA/(name+'.pdf')
        page=pymupdf.open(pdf)[0]
        page.get_pixmap(dpi=200).save(pdf.with_suffix('.png'))
    (OUT/'sweep_data_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    print('All 25 paper figures now have PDF and PNG versions.',flush=True)

if __name__ == '__main__':
    main()
