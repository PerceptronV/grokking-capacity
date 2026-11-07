#!/usr/bin/env python3
"""Relocate archived alpha-sweep explanations without re-estimating its data.

The current registry does not reproduce the published alpha-sweep inputs.
Only the trailing explanation operators are removed from the archived PDF;
all plot operators are retained byte for byte and verified as vector paths.
"""
from pathlib import Path
from collections import Counter
import io
import json
import numpy as np
import pymupdf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figure_explanations_below_20261106'
MEDIA=ROOT/'docs/papers/2026 Grokking Capacity/media/alpha_sweep/intersection'


def render():
    records=[]
    for name in ['predicted_vs_empirical','error_vs_train_fraction']:
        source=OUT/'before'/MEDIA.relative_to(ROOT)/(name+'.pdf')
        doc=pymupdf.open(source);page=doc[0]
        original_text=page.get_text();drawings=page.get_drawings()
        code=page.read_contents();scatter=name=='predicted_vs_empirical'
        marker=b'/A6 gs 0 J' if scatter else b'/A2 gs 0 J 1 g'
        assert code.count(marker)==1
        split=code.index(marker)
        frames=[(i,d) for i,d in enumerate(drawings) if d['fill'] is not None
                and d['rect'].width>20 and len(d['items'])>4]
        cut=frames[0][0]
        # The original trailing block closes the graphics state opened for the spines.
        clean=code[:split]+b'Q\n'
        xref=doc.get_new_xref();doc.update_object(xref,'<<>>');doc.update_stream(xref,clean)
        page.set_contents(xref)
        assert page.get_drawings()==drawings[:cut], 'Plot vector operators changed'
        width,height=page.rect.width,page.rect.height
        footer_height=88 if scatter else 64
        fig=plt.figure(figsize=(width/72,footer_height/72))
        if scatter:
            handles=[Line2D([],[],color='gray',ls='--',alpha=.7,label='y = x')]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.98),frameon=False,fontsize=11)
            summary=original_text[original_text.index('N ='):original_text.index('y = x')].strip()
            fig.text(.5,.57,summary,ha='center',va='top',fontsize=11,linespacing=1.3)
        else:
            markers=[d for d in drawings[cut+1:] if len(d['items'])==8 and d['fill'] is not None]
            labels=[t for t in original_text.splitlines() if t.startswith('p=')]
            assert len(markers)==len(labels)==10
            handles=[Line2D([],[],ls='',marker='o',markerfacecolor=d['fill'],markeredgecolor='black',
                            markeredgewidth=.5,markersize=np.sqrt(70),alpha=.8,label=label)
                     for d,label in zip(markers,labels)]
            fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.98),ncol=5,
                       title='P',title_fontsize=10,frameon=False,fontsize=10)
        buffer=io.BytesIO();fig.savefig(buffer,format='pdf');plt.close(fig)
        footer=pymupdf.open(stream=buffer.getvalue(),filetype='pdf')
        result=pymupdf.open();target=result.new_page(width=width,height=height+footer_height+5)
        target.show_pdf_page(pymupdf.Rect(0,0,width,height),doc,0)
        target.show_pdf_page(pymupdf.Rect(0,height+5,width,height+5+footer_height),footer,0)
        # Every original word, tick and reported statistic is retained.
        assert Counter(target.get_text().split())==Counter(original_text.split())
        pdf=MEDIA/(name+'.pdf');result.save(pdf,garbage=4,deflate=True)
        target.get_pixmap(dpi=200).save(pdf.with_suffix('.png'))
        records.append({'figure':str(pdf.relative_to(ROOT)),'method':'Relocate archived explanations; no numerical recomputation',
                        'plot_content_prefix_byte_identical':True,'plot_vector_drawings_identical':True,
                        'all_text_and_statistics_preserved':True,'preserved_plot_drawings':cut})
    return records

if __name__=='__main__':
    print(json.dumps(render(),indent=2))
