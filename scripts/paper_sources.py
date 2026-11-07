"""Find the current manuscript's TeX sources and figure assets."""
from pathlib import Path
import re


def tex_sources(paper: Path, main: str = 'paper-neurips-revised.tex') -> list[Path]:
    """Follow local input/include commands once, preserving document order."""
    seen = set()
    sources = []

    def visit(path):
        path = path.resolve()
        if path in seen:
            return
        seen.add(path)
        sources.append(path)
        text = uncomment(path.read_text())
        for name in re.findall(r'\\(?:input|include)\{([^}]+)\}', text):
            child = Path(name)
            if not child.suffix:
                child = child.with_suffix('.tex')
            visit(paper / child)

    visit(paper / main)
    return sources


def uncomment(text: str) -> str:
    """Drop TeX comments, retaining escaped percentage signs."""
    return re.sub(r'(?<!\\)%[^\n]*', '', text)


def figure_assets(paper: Path) -> list[Path]:
    """Resolve figures using the manuscript's local graphics directories."""
    assets = []
    for source in tex_sources(paper):
        text = uncomment(source.read_text())
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            candidates = [paper / name, paper / 'media' / name, paper / 'media/theory' / name]
            path = next((p for p in candidates if p.is_file()), None)
            if path is None:
                raise FileNotFoundError(f'Figure {name!r} referenced in {source}')
            assets.append(path)
    return assets
