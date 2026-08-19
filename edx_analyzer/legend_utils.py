"""Legend placement shared by every graph of the suite (EDX profile, LOM
distribution, PDF report), so a legend can be pushed completely out of the
plotting area instead of sitting on top of the curves.

`apply_legend` draws the legend, `fit_layout` gives it the room it needs.
Both take the *label* shown in the combo boxes (a key of LEGEND_POSITIONS).
"""

import math

from .constants import LEGEND_POSITIONS

MAX_COLUMNS = 5          # entries per row when the legend is above/below


def _columns(n_entries):
    return max(1, min(int(n_entries), MAX_COLUMNS))


def _clear(ax):
    """Drop any previous legend, on the axes and on the figure.

    A figure-level legend survives `ax.clear()`, so it has to be removed
    explicitly or legends would pile up on every redraw.
    """
    fig = ax.get_figure()
    for legend in list(getattr(fig, "legends", [])):
        legend.remove()
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def apply_legend(ax, label, facecolor=None, labelcolor=None, fontsize=None):
    """Place the legend of `ax` according to `label`. Returns it, or None."""
    _clear(ax)
    position = LEGEND_POSITIONS.get(label, "outside right")
    handles, labels = ax.get_legend_handles_labels()
    if position == "none" or not handles:
        return None

    args = {"frameon": True}
    if facecolor is not None:
        args["facecolor"] = facecolor
    if labelcolor is not None:
        args["labelcolor"] = labelcolor
    if fontsize is not None:
        args["fontsize"] = fontsize

    if position == "outside right":
        return ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), **args)

    if position in ("outside top", "outside bottom"):
        # A figure-level legend sits outside the axes for good: it can never
        # collide with the title, the axis labels or the zone names.
        top = position == "outside top"
        return ax.get_figure().legend(
            handles, labels,
            loc="upper center" if top else "lower center",
            bbox_to_anchor=(0.5, 0.995 if top else 0.005),
            ncol=_columns(len(labels)), **args)

    return ax.legend(handles, labels, loc=position, **args)


def _size_fraction(fig, legend, rows, horizontal):
    """Legend width/height as a fraction of the figure, measured when a
    renderer is available, estimated otherwise."""
    try:
        box = legend.get_window_extent(fig.canvas.get_renderer())
        if horizontal:
            return box.height / fig.bbox.height + 0.02
        return box.width / fig.bbox.width + 0.02
    except Exception:                                     # noqa: BLE001 - no renderer yet
        return (0.045 * rows + 0.035) if horizontal else 0.22


def fit_layout(fig, label):
    """`tight_layout()` plus the margin an outside legend needs.

    The margin is *added* to what tight_layout computed, so the legend never
    lands on the title, the axis label or the tick labels.
    """
    try:
        fig.tight_layout()
    except Exception:                                     # noqa: BLE001 - layout is best effort
        pass

    position = LEGEND_POSITIONS.get(label, "")
    if position not in ("outside right", "outside top", "outside bottom"):
        return

    params = fig.subplotpars
    legend = fig.legends[0] if fig.legends else None

    if position == "outside right":
        fig.subplots_adjust(right=min(params.right, 0.80))
        return
    if legend is None:
        return

    entries = len(legend.get_texts())
    rows = math.ceil(entries / _columns(entries)) if entries else 1
    margin = min(0.40, _size_fraction(fig, legend, rows, horizontal=True))
    if position == "outside top":
        fig.subplots_adjust(top=max(0.45, params.top - margin))
    else:
        fig.subplots_adjust(bottom=min(0.55, params.bottom + margin))
