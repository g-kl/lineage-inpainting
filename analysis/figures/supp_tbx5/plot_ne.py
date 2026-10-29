"""f to h: the neural ectoderm. reads numerical_results/umap_ne.parquet and
ne_counts.csv.

f  the NE-only umap by cell type. the progenitors are one grey, other NE, and the five
   neuron types a hue each, drawn over the progenitors.
g  the same map by dataset, shuffled, from a draw with as many cells of each.
h  neurons as a share of the NE, per embryo, the reference embryos against the crispant
   embryos, in g's dataset hues.

    python -m analysis.figures.supp_tbx5.plot_ne
"""
import os

import numpy as np
import pandas as pd

from analysis.figures.supp_tbx5 import style as S


def ne_map():
    u = pd.read_parquet(os.path.join(S.NR, "umap_ne.parquet"))
    u = u.sample(frac=1, random_state=S.SEED)
    xl, yl = S.window(u.x.values, u.y.values, S.NE_MAP, frac=0.005)
    return u, xl, yl


def one_map(u, xl, yl, w, c, name, items, kbox):
    fig, box = S.figure(w, S.ROW_H[2])
    ax = box(*S.NE_MAP)
    ax.scatter(u.x, u.y, s=S.S_UMAP, c=c, lw=0, rasterized=True)
    ax.set_xlim(*xl)
    ax.set_ylim(*yl)
    S.bare(ax)
    S.key(box(*kbox), items, ncol=2 if len(items) > 2 else 1, fs=S.FS_KEY, ms=3.0,
          gap=0.04, colgap=0.12, box=True)
    return S.save_svg(fig, name)


def render_type():
    u, xl, yl = ne_map()
    u = u.iloc[np.argsort(u.cell_type.isin(list(S.NE_NEURON)).values, kind="stable")]
    items = [("other NE", S.NE_OTHER)] + [(S.NE_SHORT.get(k, k), v)
                                          for k, v in S.NE_NEURON.items()]
    return one_map(u, xl, yl, S.NE_W, [S.NE_NEURON.get(t, S.NE_OTHER) for t in u.cell_type],
                   "f_ne_type", items, S.NE_KEY)


def render_dataset():
    u, xl, yl = ne_map()
    n = u.dataset.value_counts()
    items = [(f"{S.DATASET_NAME[k]}   $n = {n[k]:,}$", S.DATASET[k]) for k in (S.REF, S.CRI)]
    return one_map(u, xl, yl, S.NE_DS_W, [S.DATASET[k] for k in u.dataset],
                   "g_ne_dataset", items, S.NE_DS_KEY)


def render_neurons():
    c = pd.read_csv(os.path.join(S.NR, "ne_counts.csv"))
    fig, box = S.figure(S.NEURON_W, S.ROW_H[2])
    ax = box(*S.NEURON_BOX)
    rng = np.random.default_rng(S.SEED)
    for i, k in enumerate((S.REF, S.CRI)):
        g = c[c.dataset == k]
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(g)), 100 * g.share, s=14,
                   color=S.DATASET[k], lw=0, zorder=3)
        m = 100 * g.neurons.sum() / g.cells.sum()
        ax.plot([i - 0.25, i + 0.25], [m, m], color=S.INK, lw=1.0, zorder=2)
    ax.set_xticks([0, 1], [S.DATASET_NAME[k].replace(" ", "\n", 1) for k in (S.REF, S.CRI)])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, None)
    ax.set_ylabel("neurons (% of NE)")
    return S.save_svg(fig, "h_neurons")


def render():
    return render_type(), render_dataset(), render_neurons()


if __name__ == "__main__":
    print(render())
