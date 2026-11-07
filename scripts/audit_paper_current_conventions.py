#!/usr/bin/env python3
"""Read-only manuscript audit; evidence and generated previews go to reports/."""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/'docs/papers/2026 Grokking Capacity'
ENTRY=PAPER/'paper-neurips-revised.tex'
OUT=ROOT/'reports/paper_rigour_new_conventions_20261106'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=str)+'\n')
def csvsave(name,rows):
    rows=list(rows)
    if not rows:return
    with (OUT/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
def uncomment(text):return re.sub(r'(?<!\\)%[^\n]*','',text)
def rel(p):return str(Path(p).relative_to(ROOT))
def lineof(text,position):return text[:position].count('\n')+1

def inventory():
    OUT.mkdir(parents=True,exist_ok=True)
    pending=[ENTRY];seen={};missing=[]
    while pending:
        p=pending.pop(0)
        if p in seen:continue
        if not p.exists():missing.append(str(p));continue
        text=p.read_text();active=uncomment(text);seen[p]=(text,active)
        for m in re.finditer(r'\\(?:input|include)\{([^}]+)\}',active):
            target=m.group(1);target=target if Path(target).suffix else target+'.tex'
            q=PAPER/target
            if not q.exists():q=p.parent/target
            pending.append(q.resolve())
    graphics=[];tables=[];figures=[];sections=[];labels=[];references=[];citations=[];numbers=[]
    snapshots=OUT/'source_snapshot'
    for p,(raw,text) in seen.items():
        target=snapshots/p.relative_to(PAPER);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',text):
            target=m.group(1);candidates=[PAPER/target,PAPER/'media'/target,PAPER/'media/theory'/target]
            resolved=next((q for q in candidates if q.is_file()),None)
            item={'source':rel(p),'line':lineof(text,m.start()),'include':target,'resolved':rel(resolved) if resolved else None}
            if resolved:
                item.update(sha256=sha(resolved),png_exists=resolved.with_suffix('.png').exists())
                if resolved.suffix=='.pdf':
                    with pymupdf.open(resolved) as doc:
                        item.update(pages=len(doc),pdf_text='\n'.join(page.get_text() for page in doc))
            graphics.append(item)
        for env,dest in [('table',tables),('figure',figures)]:
            pat=r'\\begin\{'+env+r'\*?\}.*?\\end\{'+env+r'\*?\}'
            for m in re.finditer(pat,text,re.S):
                block=m.group(0);labs=re.findall(r'\\label\{([^}]+)\}',block)
                dest.append({'source':rel(p),'line':lineof(text,m.start()),'labels':labs,'text':block})
        for m in re.finditer(r'\\(?:sub)*section\*?\{([^}]+)\}',text):sections.append({'source':rel(p),'line':lineof(text,m.start()),'heading':m.group(1)})
        for m in re.finditer(r'\\label\{([^}]+)\}',text):labels.append({'name':m.group(1),'source':rel(p),'line':lineof(text,m.start())})
        for m in re.finditer(r'\\(?:[cC]ref|ref|eqref|autoref|pageref)\{([^}]+)\}',text):
            references.extend({'name':k.strip(),'source':rel(p),'line':lineof(text,m.start())} for k in m.group(1).split(','))
        for m in re.finditer(r'\\cite\w*\*?(?:\[[^\]]*\])*\{([^}]+)\}',text):
            citations.extend({'name':k.strip(),'source':rel(p),'line':lineof(text,m.start())} for k in m.group(1).split(','))
        for i,line in enumerate(text.splitlines(),1):
            if re.search(r'\d',line):numbers.append({'source':rel(p),'line':i,'text':line})
    bib=PAPER/'references.bib';bibkeys=re.findall(r'@\w+\s*\{\s*([^,]+),',bib.read_text())
    names=Counter(x['name'] for x in labels)
    manifest={'entry':rel(ENTRY),'included_sources':[{'path':rel(p),'sha256':sha(p),'lines':len(t.splitlines())} for p,(t,_) in seen.items()],
        'missing_inputs':missing,'graphic_count':len(graphics),'table_count':len(tables),'figure_environment_count':len(figures),'section_count':len(sections),
        'missing_graphics':[g for g in graphics if not g['resolved']],
        'graphics_without_png':[g['resolved'] for g in graphics if g.get('resolved') and not g.get('png_exists')],
        'duplicate_labels':{k:v for k,v in names.items() if v>1},
        'undefined_references':[r for r in references if r['name'] not in names],
        'undefined_citations':[r for r in citations if r['name'] not in bibkeys],
        'duplicate_bibkeys':{k:v for k,v in Counter(bibkeys).items() if v>1},
        'cited_keys':sorted({r['name'] for r in citations}),
        'paper_pdf_sha256':sha(ENTRY.with_suffix('.pdf')),
        'all_input_hashes':{**{rel(p):sha(p) for p in seen},rel(bib):sha(bib),**{g['resolved']:g['sha256'] for g in graphics if g['resolved']}}}
    save('inventory.json',manifest);save('graphics.json',graphics);save('tables.json',tables);save('figure_environments.json',figures)
    csvsave('sections.csv',sections);csvsave('numeric_text_inventory.csv',numbers);csvsave('citations.csv',citations)
    with pymupdf.open(ENTRY.with_suffix('.pdf')) as doc:
        save('paper_pdf_pages.json',[{'page':i+1,'width':p.rect.width,'height':p.rect.height,'text':p.get_text()} for i,p in enumerate(doc)])
    requirements={
        'objective':'Do another rigour and consistency check over the paper text, numbers, tables, figures, data, and appendices, bearing in mind our new conventions.',
        'conventions':{'laws':'Fit every uncensored seed individually by OLS in log time for BOTH clocks; no seed aggregation before fitting.',
            'onset':'Minimum paired nonnegative delay over ten seeds 42–51; fixed width-domain last-zero/positive-tail rule.',
            'thresholds':'Clock fits 99%; delay training99 and validation98, canonical online training mode.',
            'fit_domain':'Ten primes 97,101,103,107,109,113,127,131,137,139; fixed original f window0.08–1 and width<=256.',
            'censoring':'Exclude unobserved events from clock fits; do not use caps as observed crossings; retain unresolved cone outcomes.',
            'hardware':'A100-only central clock fits; H100 matched capacity; cone cells contain one hardware cohort and ten seeds each.',
            'figures':'PDF for paper, PNG preview; no titles; explanations below; dotted extrapolation beyond139; no blue shading.'},
        'coverage':['active main text and included appendices/checklist','every active table','every included figure and PDF/PNG parity','current coefficients and all dependent forecasts/statistics','raw trace events and seed/hardware provenance','mechanistic and algebraic claims','build, internal references and bibliography consistency'],
        'deliverables':['evidence-backed findings with exact locations','coverage ledger with supported, contradicted, unverifiable states','recomputed numerical comparisons','figure/source/build evidence','concise prioritized report'],
        'scope':'Audit and reproducibility checks. No paper wording or numerical replacements made during the audit; findings state required changes.'}
    save('requirements.json',requirements)
    print(json.dumps({k:manifest[k] for k in ['included_sources','graphic_count','table_count','figure_environment_count','section_count','missing_inputs','missing_graphics','graphics_without_png','duplicate_labels','undefined_references','undefined_citations']},indent=2))

if __name__=='__main__':inventory()
