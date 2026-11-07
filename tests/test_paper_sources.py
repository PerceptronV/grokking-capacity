"""Publication builds follow the current manuscript, not stale audit inventories."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from paper_sources import figure_assets, tex_sources


def test_nested_figures_and_repeated_includes(tmp_path):
    (tmp_path / 'theory').mkdir()
    (tmp_path / 'media').mkdir()
    (tmp_path / 'paper-neurips-revised.tex').write_text(
        r'\input{theory/section}\input{theory/section}' + '\n'
        r'% \input{missing}' + '\n'
        r'\includegraphics[width=.8\linewidth]{media/direct.pdf}'
    )
    (tmp_path / 'theory/section.tex').write_text(r'\input{theory/nested.tex}')
    (tmp_path / 'theory/nested.tex').write_text(
        r'\includegraphics{nested.pdf}' + '\n' + r'% \includegraphics{missing.pdf}'
    )
    for name in ['direct.pdf', 'nested.pdf']:
        (tmp_path / 'media' / name).touch()
    assert len(tex_sources(tmp_path)) == 3
    assert {p.name for p in figure_assets(tmp_path)} == {'direct.pdf', 'nested.pdf'}


def test_missing_figure_reports_its_source(tmp_path):
    (tmp_path / 'paper-neurips-revised.tex').write_text(r'\includegraphics{absent.pdf}')
    with pytest.raises(FileNotFoundError, match='absent.pdf.*paper-neurips-revised.tex'):
        figure_assets(tmp_path)
