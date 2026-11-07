#!/usr/bin/env python3
"""Validate publication tables, numerical roots, figures, provenance and build."""
import ast,json,re,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import pymupdf as fitz
from PIL import Image,ImageDraw
from publish_paper_revision import ROOT,PAPER,REPORT,RESULTS,SOURCES,LABELS,AXES,PUBLISHED_CROSSING_PROTOCOLS,pooled,within_table,crossing_table,sha,save_json
from paper_tables import threshold_table
from paper_sources import figure_assets
from recompute_paper_statistics import prepare
from grokking_capacity.analysis.aggregate import intersection_summary


def main():
    s=(PAPER/'paper-neurips-revised.tex').read_text();checks={};tables=0
    within=pd.read_csv(RESULTS/'statistics/within_setting.csv')
    for name in LABELS:
        test=json.loads((RESULTS/'statistics'/name/'hypothesis_tests.json').read_text())
        assert pooled(name,test) in s;tables+=1
    for name in AXES:assert within_table(name,within) in s;tables+=1
    for which in PUBLISHED_CROSSING_PROTOCOLS:
        cross=pd.read_csv(RESULTS/'crossings'/f'crossing_{which}.csv');cells=pd.read_csv(RESULTS/'crossings'/f'cells_{which}.csv')
        assert crossing_table(which,cross) in s;tables+=1
        for _,row in cells.iterrows():
            for kind in ['mem','gen']:
                values=[v for v in ast.literal_eval(row[kind+'_times']) if v is not None]
                mean=np.mean(values) if values else np.nan
                np.testing.assert_allclose(row[kind+'_mean'],mean,equal_nan=True)
        for row in cross.itertuples():
            cc=cells[cells.p==row.p];curves=[]
            for key in ['mem_mean','gen_mean']:
                curves.append({int(r.P):float(getattr(r,key)) for r in cc.itertuples() if np.isfinite(getattr(r,key))})
            root=intersection_summary(*curves,direction='decreasing')
            assert root['status']==row.crossing_status
            np.testing.assert_allclose(np.log10(root['value']),row.crossing_log10P,rtol=1e-12)
        checks[which]={'roots':int(cross.crossing_log10P.notna().sum()),'onsets':int(cross.onset_log10P.notna().sum()),'agreement_matches':int(cross.agreement_matches.sum()),'scored_cells':int(cross.scored_cells.sum()),'all_cells':len(cells)}
    assert threshold_table(json.loads((RESULTS/'supplement/sensitivities.json').read_text())) in s;tables+=1
    log=(REPORT/'build/paper-neurips-revised.log').read_text()
    assert 'Overfull \\hbox' not in log
    assert not re.search(r'(?:LaTeX|Package [^\n]+) Warning:.*(?:undefined|multiply defined)',log)
    for name in SOURCES:
        ss=(PAPER/name).read_text()
        assert '167.5' not in ss and 'unrecoverable' not in ss.lower()
        assert not re.search(r'\bmup\b|per[-\s]+layer|(?-i:μP|µP)',ss,re.I),name
        # TeX structure, including the nested table/figure environments.
        stack=[]
        for m in re.finditer(r'\\(begin|end)\{([^}]+)\}',ss):
            if m[1]=='begin':stack.append(m[2])
            else:assert stack and stack.pop()==m[2],(name,m.group())
        assert not stack,(name,stack)
    graphics=[]
    for p in figure_assets(PAPER):
        assert p.suffix=='.pdf' and p.with_suffix('.png').exists()
        graphics.append({'path':str(p.relative_to(ROOT)),'pdf_sha256':sha(p),'png_sha256':sha(p.with_suffix('.png'))})
    assert len(graphics)==23+len(PUBLISHED_CROSSING_PROTOCOLS)
    # Raw-file fingerprints accompany every selected analysis membership.
    _,pool,_,_,_=prepare()
    def fingerprint(row):return {'uuid':row['uuid'],'path':row['npz_path'],'sha256':sha(row['npz_path'])}
    with ThreadPoolExecutor(max_workers=8) as workers:traces=list(workers.map(fingerprint,pool.values()))
    pd.DataFrame(traces).to_csv(RESULTS/'statistics/trace_manifest.csv',index=False)
    print('Verified tables, roots and',len(traces),'trace hashes',flush=True)
    # Render the complete final paper for layout review and archival provenance.
    out=REPORT/'renders';out.mkdir(exist_ok=True);doc=fitz.open(PAPER/'paper-neurips-revised.pdf')
    assert not re.search(r'\bmup\b|per[-\s]+layer|(?-i:μP|µP)','\n'.join(page.get_text() for page in doc),re.I)
    for offset in range(0,len(doc),12):
        sheet=Image.new('RGB',(1224,1260),'#dadada');draw=ImageDraw.Draw(sheet)
        for j in range(offset,min(offset+12,len(doc))):
            pix=doc[j].get_pixmap(matrix=fitz.Matrix(.46,.46));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((296,392));x=((j-offset)%4)*306+5;y=((j-offset)//4)*420+22;sheet.paste(im,(x,y));draw.text((x,y-17),str(j+1),fill='black')
        sheet.save(out/f'pages_{offset+1:02d}.png')
    for offset in range(0,len(graphics),9):
        sheet=Image.new('RGB',(1500,1200),'#dadada');draw=ImageDraw.Draw(sheet)
        for j in range(offset,min(offset+9,len(graphics))):
            im=Image.open((ROOT/graphics[j]['path']).with_suffix('.png')).convert('RGB');im.thumbnail((490,365));x=((j-offset)%3)*500+5;y=((j-offset)//3)*400+28;sheet.paste(im,(x,y));draw.text((x,y-20),str(j+1),fill='black')
        sheet.save(out/f'figures_{offset+1:02d}.png')
    models=json.loads((RESULTS/'models/models.json').read_text())
    checks.update({'published_crossing_protocols':list(PUBLISHED_CROSSING_PROTOCOLS),'generated_tables_checked':tables,'pdf_figures_with_png':len(graphics),'raw_traces_hashed':len(traces),'pages':len(doc),'no_overfull_boxes':True,'no_unresolved_references':True,'law_verification':models['verification'],'sources':{n:sha(PAPER/n) for n in SOURCES},'graphics':graphics,'paper_pdf_sha256':sha(PAPER/'paper-neurips-revised.pdf'),'validation_source_sha256':sha(__file__)})
    save_json(REPORT/'publication_validation.json',checks)
    man=json.loads((REPORT/'publication_manifest.json').read_text());man.update(status='published and numerically validated; author submission tasks documented separately',validation=str((REPORT/'publication_validation.json').relative_to(ROOT)),paper_pdf_sha256=checks['paper_pdf_sha256']);save_json(REPORT/'publication_manifest.json',man)
    latest=ROOT/'results/cone_latest.json';backup=REPORT/'cone_latest_before.json'
    if not backup.exists():shutil.copy2(latest,backup)
    old=json.loads(latest.read_text());old.update(analysis_directory='results/paper_revision_20261107/models',figure_pdf='results/paper_revision_20261107/models/cone_min_cellmeans.pdf',figure_png='results/paper_revision_20261107/models/cone_min_cellmeans.png',publication_validation=str((REPORT/'publication_validation.json').relative_to(ROOT)),published_utc=datetime.now(timezone.utc).isoformat(),clock_aggregation='arithmetic observed cell means; log-OLS with equal cell weights',fit='results/paper_revision_20261107/models/models.json',publication_command='python scripts/build_revised_paper.py --reuse-analyses')
    save_json(latest,old)
    print('Publication validation passed:',tables,'tables,',len(graphics),'figure pairs,',len(doc),'pages.',flush=True)


if __name__=='__main__':main()
