#!/usr/bin/env python3
"""Update validated numbers, the decay plot, and approved norm-freezing wording."""
from pathlib import Path
import hashlib,json,re,shutil
ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
CAMPAIGN=ROOT/'results/normfrozen_h100_paired_20261103'
WD=ROOT/'results/weight_decay_matched_20261103'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    wd=json.loads((WD/'summary.json').read_text());n=json.loads((CAMPAIGN/'analysis.json').read_text());before=json.loads((CAMPAIGN/'publication_before_hashes.json').read_text())
    assert wd['selected_runs']==400 and wd['primary']['runs']==300 and wd['censored_runs']==12
    assert n['pairs_verified']==15 and len(n['checkpoint_checks'])==30 and n['censored_runs']==0
    decay=f"{wd['primary']['slope']:.2f}";frozen=f"{n['frozen']['lambda_slope_d128']:.2f}";unfrozen=f"{n['unfrozen']['lambda_slope_d128']:.2f}"
    patches=[]
    for filename,expected in [('theory/extra.tex',4),('theory/principled.tex',1)]:
        p=PAPER/filename;old=(CAMPAIGN/'before'/p.relative_to(ROOT)).read_text();new=old
        assert old.count('-0.42')==expected
        new=new.replace('-0.42',decay)
        if filename=='theory/principled.tex':
            assert new.count('$0.42$ measured')==1;new=new.replace('$0.42$ measured',f'${abs(wd["primary"]["slope"]):.2f}$ measured')
            token='($\\lambda$-slope $-0.61$ vs $-0.69$ unfrozen)'
            assert new.count(token)==1;new=new.replace(token,f'($\\lambda$-slope ${frozen}$ vs ${unfrozen}$ unfrozen)')
        # Preserve every nonnumeric character in the manuscript.
        mask=lambda text:re.sub(r'\d+(?:\.\d+)?','<NUMBER>',text)
        assert mask(new)==mask(old),'Wording would change'
        numeric_new=new
        if filename=='theory/principled.tex':
            # User approved this wording on 2026-11-04 after the paired analysis.
            old_item=f'''\\item \\emph{{Norm-gain freezing is a passed consistency check, not the
  falsifier.}} Freezing the RMSNorm gains leaves the clock unchanged
  ($\\lambda$-slope ${frozen}$ vs ${unfrozen}$ unfrozen) — as it must, since
  scale-invariance of the pre-norm weights survives gain freezing. The
  sharp falsifier is architectural (remove the normalisation operation);
  it remains open.
'''
            new_item=f'''\\item \\emph{{Norm-gain freezing.}} Freezing the RMSNorm gains preserves
  the qualitative decay dependence, while modestly changing generalisation
  times and the fitted decay exponent (${frozen}$ frozen versus ${unfrozen}$
  unfrozen). Removing the normalisation operation remains untested.
'''
            assert new.count(old_item)==1;new=new.replace(old_item,new_item)
        current=p.read_text();assert current in [old,numeric_new,new],f'Unrelated edits in {p}; preserve them before publishing'
        p.write_text(new);patches.append({'path':str(p.relative_to(ROOT)),'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':sha(p),'nonnumeric_text_identical':mask(new)==mask(old),'approved_norm_freezing_wording':filename=='theory/principled.tex'})
    for dest in [ROOT/'figures/theory/tgen_lambda.pdf',PAPER/'media/theory/tgen_lambda.pdf']:
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(WD/'tgen_lambda.pdf',dest)
    shutil.copy2(WD/'summary.json',ROOT/'results/theory/weight_decay_scaling.json')
    # The cone and main table/source remain exactly as at the start of this task.
    unchanged=['docs/papers/2026 Grokking Capacity/paper-neurips-revised.tex','docs/papers/2026 Grokking Capacity/media/theory/cone_seedmin_fixed.pdf']
    assert all(sha(ROOT/rel)==before[rel] for rel in unchanged)
    delta=n['paired_effect'];flags=[
      'The decay estimate now uses twelve fully observed widths and five paired seeds per setting. Four censored width groups are retained in the audit and restricted-time sensitivity. This selection detail belongs in the methods when wording edits are authorised.',
      'The exact seed bootstrap is conditional on the fixed grid, with five seeds for decay and three for norm freezing. It is not a population-of-widths or formal equivalence analysis.',
      f'Resolved with user-approved wording on 2026-11-04: replaced "leaves the clock unchanged" with preservation of qualitative decay dependence and explicit timing/exponent changes. The paired evidence is: the paired lambda-slope difference is {delta["lambda_slope_difference"]:+.3f}, with descriptive three-seed interval {delta["seed_bootstrap_CI95"][0]}. The fixed-grid interval excludes zero; it is not a formal equivalence test. The width betas also differ ({n["frozen"]["beta_lambda1"]:.3f} frozen versus {n["unfrozen"]["beta_lambda1"]:.3f} unfrozen). This does not establish that removing normalisation has the same effect.',
      'The norm intervention freezes gains; it does not remove the RMS normalisation operation. The existing statement that removing normalisation remains a separate test is preserved.',
      'The weight-decay experiment here concerns the appendix T_gen-versus-decay scaling. It does not alter the main onset/crossing robustness table or the central/fold predictions.',
      'The cone issue is deferred as requested; its file and the main manuscript source remain unchanged.'
    ]
    result={'weight_decay_primary':wd['primary'],'weight_decay_sensitivity':wd['sensitivity'],'frozen_slope':n['frozen']['lambda_slope_d128'],'unfrozen_slope':n['unfrozen']['lambda_slope_d128'],'paired_effect':delta,'source_edits':patches,'unchanged_main_source_and_cone':True,'flags':flags}
    (CAMPAIGN/'publication_update.json').write_text(json.dumps(result,indent=2)+'\n')
    report=f'''MATCHED WEIGHT-DECAY AND NORM-FREEZING CORRECTIONS

Weight decay
400 archived A100 runs: p113, division, train fraction .5, depth2, heads1,
dropout .2, init scale1, lr .001, Adam betas .9/.98, batch512, 5000-epoch
budget, five identical seeds 42–46 at every width/decay value.
12 runs are right-censored across widths20/24/28/32. The primary estimate
uses all 300 runs at the other twelve widths40–256: no outcome is dropped
within an included cell. Median slope = {wd['primary']['slope']:.9f}; exact
paired-seed bootstrap 95% interval = {wd['primary']['seed_bootstrap_CI95']}.
The full 16-width restricted-mean-to-5000 sensitivity is
{wd['sensitivity']['all_widths_restricted_mean_to_5000_slope']:.9f}. Its estimand is
min(T,5000), not unrestricted first-passage time. All 400 source traces were
checked against the registry, including their censoring status. The old broad
query was removed from make_theory_figures.py; it now calls the strict analysis.

Norm freezing
Thirty new runs: five settings x three seeds42–44 x two arms. Both arms ran
on host {n['host']}, GPU {n['gpu_uuid']}, with the same source snapshot and
runtime. Each pair has identical data, initial weights, and CPU/CUDA random
states. Frozen gains remain unchanged and unfrozen gains change. All30 final
checkpoints were independently evaluated on CPU; all first-val99 crossings
were independently recomputed from traces. There are no censored runs.
Frozen lambda slope: {n['frozen']['lambda_slope_d128']:.9f}
Unfrozen lambda slope: {n['unfrozen']['lambda_slope_d128']:.9f}
Paired difference: {delta['lambda_slope_difference']:.9f}
Descriptive paired-seed 95% interval: {delta['seed_bootstrap_CI95'][0]}
Frozen width beta: {n['frozen']['beta_lambda1']:.9f}
Unfrozen width beta: {n['unfrozen']['beta_lambda1']:.9f}
Width fits use the same total architecture parameter count for both arms;
trainable counts are retained separately. The analysis now requires paired
campaign manifests and never obtains controls from the mixed root registry.

Publication
The revised paper's included extra.tex changes only numerical tokens.
principled.tex also includes the user-approved norm-gain-freezing wording
(2026-11-04), acknowledging changes in timing and the fitted decay exponent. The existing tgen_lambda.pdf asset is regenerated. New diagnostic
plots are in the two campaign directories. The cone file and the main source
(including the main test table) are byte-identical to the starting versions.

Wording notes and remaining flags
'''+''.join(f'{i+1}. {s}\n' for i,s in enumerate(flags))+'''
Reproduction
python scripts/weight_decay_scaling.py
python scripts/run_normfrozen_h100_paired.py
python scripts/normfrozen_analysis.py
python scripts/update_matched_mechanisms_paper.py

The paired controller resumes completed jobs, refuses a changed training-source
snapshot, isolates frozen/unfrozen registries, and uses GPU0. Training source,
per-job provenance, dataset/initialisation/RNG hashes, raw curves, checkpoints,
analyses, and the publication snapshot are retained under the campaign directory.
'''
    (CAMPAIGN/'corrections.txt').write_text(report)
    print(json.dumps({k:result[k] for k in ['frozen_slope','unfrozen_slope','paired_effect','unchanged_main_source_and_cone']},indent=2))
if __name__=='__main__':main()
