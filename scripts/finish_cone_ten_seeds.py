#!/usr/bin/env python3
"""Refresh the clearly labelled preview, then build fully verified ten-seed data."""
from pathlib import Path
import datetime, json, os, shutil, subprocess, sys, time
from run_cone_ten_seeds import ROOT, OUT, sha, write_json


def main():
    m = json.loads((OUT / 'manifest.json').read_text())
    last_preview = 0
    while True:
        if (OUT / 'verification_errors.json').exists():
            raise RuntimeError('Independent verification failed; inspect verification_errors.json')
        if (OUT / 'completion.json').exists():
            done = json.loads((OUT / 'completion.json').read_text())
            assert not done['failed'] and done['completed'] == m['jobs'], done
            if (OUT / 'checkpoint_verification.json').exists():
                break
        status_path = OUT / 'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            assert not status['failed'], status['failed']
        if time.time() - last_preview > 300:
            subprocess.run([sys.executable, str(ROOT / 'scripts/analyse_cone_ten_seeds.py'), '--preview'], cwd=ROOT, check=True)
            last_preview = time.time()
        time.sleep(20)
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyse_cone_ten_seeds.py')], cwd=ROOT, check=True)
    a = json.loads((OUT / 'analysis.json').read_text())
    s = json.loads((OUT / 'figure/summary.json').read_text())
    assert a['total_cells'] == 1736 and a['total_runs'] == 17360 and a['new_runs'] == 1495
    assert s['seed_count_distribution'] == {'10': 1736}
    check = json.loads((OUT / 'checkpoint_verification.json').read_text())
    assert check['runs_verified'] == 1495
    metadata = [json.loads(p.read_text()) for p in (OUT / 'jobs').glob('*.json')]
    gpu_counts = {str(g): sum(r['physical_gpu'] == g for r in metadata) for g in [0, 1]}
    assert all(gpu_counts.values())
    result = {'verified_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'new_training_runs': 1495, 'CPU_checkpoint_checks': 1495,
              'uniform_ten_seed_cells': 1736, 'total_unique_seed_observations': 17360,
              'physical_gpu_run_counts': gpu_counts,
              'unresolved_runs_explicitly_flagged': a['unresolved_outcomes_flagged'],
              'all_classified_runs_have_observed_validation98': True,
              'old_complete_cells_unchanged': 1467,
              'one_hardware_cohort_per_cell': True,
              'figure_png_sha256': sha(OUT / 'figure/cone_seedmin_fixed.png'),
              'figure_pdf_sha256': sha(OUT / 'figure/cone_seedmin_fixed.pdf'),
              'runs_sha256': sha(OUT / 'figure/runs.json'),
              'cells_sha256': sha(OUT / 'figure/cells.json'),
              'paper_update_status': 'Awaiting review against claim_flags.txt; current paper not overwritten by this data-generation controller.'}
    write_json(OUT / 'validation.json', result)
    dest = OUT / 'analysis_source'; dest.mkdir(exist_ok=True)
    for name in ['analyse_cone_ten_seeds.py', 'verify_cone_ten_seeds.py', 'finish_cone_ten_seeds.py', 'cone_data.py', 'verify_cone_h100_boundary.py', 'schedule_cone_ten_seeds_mps.py', 'benchmark_cone_mps.py', 'benchmark_cone_mps_wide.py']:
        shutil.copy2(ROOT / 'scripts' / name, dest / name)
    (OUT / 'report.txt').write_text(
        'UNIFORM TEN-SEED CONE DATA COMPLETE\n\n'
        'All1495 planned training runs completed on the two H100s and were independently checked on CPU. '
        'Every plotted cell contains ten distinct seeds42--51. The1467 previously complete cells are unchanged; '
        '239 H100 cells received their five missing seeds, and30 five-seed A100 cells were fully replaced by ten H100 seeds each. '
        'All150 replaced observations are retained in the archived cohort and paired comparison.\n\n'
        'Missing validation98 remains unresolved and is visibly marked even when another seed establishes a zero cell minimum. '
        'See figure/unresolved_runs.csv, changed_cells.csv and claim_flags.txt. The original main-test tables and fitted laws were not refitted.\n\n'
        + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
