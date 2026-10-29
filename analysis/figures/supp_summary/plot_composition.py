"""panel a: cell type map edges by category, with the category legend. panel b: single cell
transitions by the category of their edge, in percent. reads numerical_results/transitions.csv."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from analysis.figures.supp_summary import style as S

BAR_H = 0.5
MIN_LABEL = 0.04   # segments narrower than this share of the bar carry no label


def counts():
    """edges and cells per category, identity ones in `match`."""
    t = pd.read_csv(os.path.join(S.NR, "transitions.csv"))
    n = t.category.value_counts().reindex(S.CATEGORIES, fill_value=0)
    cells = t.groupby("category")["count"].sum().reindex(S.CATEGORIES, fill_value=0)
    return n, cells


def _bar(name, n, fmt, xlabel, legend):
    S.apply_rc()
    h = S.BAR_BOTTOM + S.BAR_AX_H + (S.LEG_H if legend else S.BAR_TOP)
    fig, ax = plt.subplots(figsize=(S.PAGE_W_IN, h))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=S.BAR_BOTTOM / h,
                        top=(S.BAR_BOTTOM + S.BAR_AX_H) / h)
    left = 0
    for v in S.CATEGORIES:
        ax.barh(0, n[v], left=left, height=BAR_H, color=S.CATEGORY_COLORS[v],
                edgecolor="white", linewidth=0.8)
        if n[v] >= MIN_LABEL * n.sum():
            ax.text(left + n[v] / 2, 0, fmt(n[v]), ha="center", va="center")
        left += n[v]
    ax.set_ylim(-BAR_H / 2 - 0.05, BAR_H / 2 + 0.05)
    ax.set_xlim(0, n.sum() * 1.01)
    ax.set_xlabel(xlabel)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    if legend:
        fig.legend(handles=[Patch(color=S.CATEGORY_COLORS[v], label=S.CATEGORY_LABELS[v])
                            for v in S.CATEGORIES],
                   loc="upper right", bbox_to_anchor=(0.98, 1.0), ncol=len(S.CATEGORIES),
                   frameon=False, handlelength=1.0, handleheight=0.8, handletextpad=0.4,
                   columnspacing=1.1, borderaxespad=0)
    out = os.path.join(S.PANELS, name)
    fig.savefig(out)
    plt.close(fig)
    return out, (S.PAGE_W_IN, h)


def render():
    n, _ = counts()
    return _bar("a_composition.svg", n, str, "cell type map edges", True)


def render_cells():
    _, cells = counts()
    return _bar("b_cells.svg", 100 * cells / cells.sum(), lambda v: f"{v:.0f}%",
                "single cell transitions (%)", False)
