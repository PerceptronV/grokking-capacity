#!/usr/bin/env python3
"""Remove only the two headings from the preserved legacy vector figure."""
from pathlib import Path
from collections import Counter
import json,math
import numpy as np
import pymupdf
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figure_titles_removed_20261106'
REL=Path('docs/papers/2026 Grokking Capacity/media/grokking_separate_p127.pdf')
source=OUT/'before'/REL;target=ROOT/REL
before=pymupdf.open(source);doc=pymupdf.open(source);page=doc[0]
titles=['Training Accuracy Curves','Validation Accuracy Curves'];rects=[]
for title in titles:
    found=page.search_for(title);assert len(found)==1,(title,found)
    rect=found[0];rects.append(rect);page.add_redact_annot(rect,fill=False,cross_out=False)
page.apply_redactions(images=0,graphics=0,text=0)
assert all(title not in page.get_text() for title in titles)
assert Counter(page.get_text().split())==Counter(before[0].get_text().split())-Counter(' '.join(titles).split())
# Removing text changes PDF operator sequence numbers, not the drawn geometry.
old_paths=before[0].get_drawings();new_paths=page.get_drawings()
assert len(old_paths)==len(new_paths)
assert all({k:v for k,v in a.items() if k!='seqno'}=={k:v for k,v in b.items() if k!='seqno'} for a,b in zip(old_paths,new_paths))
scale=2;old=before[0].get_pixmap(matrix=pymupdf.Matrix(scale,scale));new=page.get_pixmap(matrix=pymupdf.Matrix(scale,scale))
a=np.frombuffer(old.samples,np.uint8).reshape(old.height,old.width,old.n);b=np.frombuffer(new.samples,np.uint8).reshape(new.height,new.width,new.n)
mask=np.zeros(a.shape[:2],bool)
for r in rects:
    mask[max(0,math.floor(r.y0*scale)-3):math.ceil(r.y1*scale)+3,max(0,math.floor(r.x0*scale)-3):math.ceil(r.x1*scale)+3]=True
assert not np.any(np.any(a!=b,axis=2)&~mask),'Content outside title areas changed'
tmp=target.with_suffix('.notitles.tmp.pdf');doc.save(tmp,garbage=4,deflate=True);doc.close();tmp.replace(target)
result={'titles_removed':titles,'all_other_text_preserved':True,'all_vector_drawings_preserved':True,'pixels_outside_title_regions_identical':True}
(OUT/'legacy_plot_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
