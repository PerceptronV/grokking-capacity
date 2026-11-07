"""Paper figure layout: explanations below the axes, PDF and PNG exports."""
from pathlib import Path


def explanations_below(fig, *, notes=(), max_columns=3):
    """Move legends and fit-summary boxes to a footer without changing data artists.

    Numeric labels attached to points/axes and colour scales stay in place.
    Explicit ``notes`` also supports summaries without a background box.
    """
    if getattr(fig, '_explanations_below', False):
        return
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    legends += list(fig.legends)
    summaries = list(notes)
    for ax in fig.axes:
        for text in ax.texts:
            if text.get_bbox_patch() is not None and text not in summaries:
                # Prime identifiers identify panels, rather than explaining a fit.
                if not text.get_text().startswith('$p='):
                    summaries.append(text)
    entries = []
    for legend in legends:
        entries.append((list(legend.legend_handles),
                        [text.get_text() for text in legend.get_texts()],
                        legend.get_texts()[0].get_fontsize(),
                        legend.get_title().get_text()))
        legend.remove()
    note_values = [(text.get_text(), text.get_fontsize()) for text in summaries]
    for text in summaries:
        text.remove()
    fig.canvas.draw()
    fig.set_layout_engine(None)
    y = -0.015
    for handles, labels, fontsize, title in entries:
        for columns in range(min(max_columns, len(labels)), 0, -1):
            legend = fig.legend(handles, labels, loc='upper center',
                                bbox_to_anchor=(0.5, y), bbox_transform=fig.transFigure,
                                ncol=columns, fontsize=fontsize, frameon=False,
                                title=title or None, title_fontsize=fontsize,
                                borderaxespad=0, columnspacing=1.4, handlelength=2.3)
            fig.canvas.draw()
            bounds = legend.get_window_extent().transformed(fig.transFigure.inverted())
            if bounds.width <= .96 or columns == 1:
                break
            legend.remove()
        y = bounds.y0 - .025
    for value, fontsize in note_values:
        note = fig.text(.5, y, value, ha='center', va='top', fontsize=fontsize,
                        transform=fig.transFigure, linespacing=1.3)
        fig.canvas.draw()
        bounds = note.get_window_extent().transformed(fig.transFigure.inverted())
        y = bounds.y0 - .025
    fig._explanations_below = True


def save_paper_figure(fig, path, *, notes=(), max_columns=3, dpi=200, footer=True):
    """Save PDF/PNG; footer=False preserves legends inside the axes."""
    if footer:
        explanations_below(fig, notes=notes, max_columns=max_columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png'):
        fig.savefig(path.with_suffix(suffix), bbox_inches='tight', dpi=dpi)
