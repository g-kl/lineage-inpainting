"""Commitment-signature query layer.

The index (built once by build_index.py) holds, per assigned internal node in
the 9.5-leaf-time trees: its tree, node, assigned cell, time, time bin, total
descendant-leaf count, and a descendant-leaf label histogram at each level
(lineage, cell_type). Plus a cached expression matrix for every assigned cell.

Everything a query needs derives from those histograms: pair-relative
categories (only_A / only_B / both / neither), global commitment, and DEGs.
Loading trees and reading expression -- the slow, label-independent work -- is
done once at build time, so queries here run in seconds.
"""

import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

# paths are absolute (resolved from this file) so queries work from any cwd
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "..", "data", "embryos")
H5AD = os.path.join(DATA_DIR, "qiu24_integrated_hvgs_and_scvi.h5ad")
TREES = os.path.join(DATA_DIR, "final_topology.h5td")
TYPE_INFO = os.path.join(DATA_DIR, "final_type_info_v2_lower.csv")
SUBTYPES = os.path.join(DATA_DIR, "final_subtype_v2_lower.csv")
ASSIGN_PKL = os.path.join(
    REPO_ROOT,
    "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none"
    "_57ab441ec3fcecf81d545755db5ddb09/"
    "assignments_nn_mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none"
    "_57ab441ec3fcecf81d545755db5ddb09_step20000_adj_n1.pkl",
)

INDEX_DIR = os.path.join(REPO_ROOT, "analysis/pipelines/commitment/index")
LEVELS = ["lineage", "cell_type"]
LEAF_TIME = 9.5
BIN_EDGES = np.round(np.arange(7.5, 9.5 + 1e-9, 0.25), 2)
NBIN = len(BIN_EDGES) - 1
BIN_LABELS = [f"{BIN_EDGES[i]:.2f}-{BIN_EDGES[i + 1]:.2f}" for i in range(NBIN)]
UNKNOWN = "unknown"
EPS = 1e-6


def bin_of(t):
    """Index of the quarter-Myr bin holding time t, or -1 if outside range."""
    if t < BIN_EDGES[0] - EPS or t >= BIN_EDGES[-1] - EPS:
        return -1
    return int(np.floor((t - BIN_EDGES[0]) / 0.25 + EPS))


class Index:
    """Loaded commitment index. Attributes:

    nodes : DataFrame (tree, node, cell, time, bin, n_leaf)
    hist  : {level: int array (n_nodes, n_categories)}
    cols  : {level: list of category names}
    expr  : csr matrix (n_cells, n_genes)
    genes : array of gene names
    row_of: {cell name: row in expr}
    """

    def __init__(self, index_dir=INDEX_DIR):
        self.nodes = pd.read_parquet(os.path.join(index_dir, "nodes.parquet"))
        self.hist, self.cols = {}, {}
        for lv in LEVELS:
            self.hist[lv] = np.load(os.path.join(index_dir, f"hist_{lv}.npy"))
            self.cols[lv] = list(
                np.load(os.path.join(index_dir, f"cols_{lv}.npy"), allow_pickle=True)
            )
        self.expr = sp.load_npz(os.path.join(index_dir, "expr.npz"))
        self.genes = np.load(os.path.join(index_dir, "genes.npy"), allow_pickle=True)
        cells = np.load(os.path.join(index_dir, "expr_cells.npy"), allow_pickle=True)
        self.row_of = {c: i for i, c in enumerate(cells)}

    def fraction(self, level, label):
        """Per-node fraction of descendant leaves carrying `label`.

        `label=None` means the complement (any leaf labeled at this level but
        not equal to the query's A label); handled by the caller.
        """
        H = self.hist[level]
        j = self.cols[level].index(label)
        return H[:, j] / self.nodes["n_leaf"].values

    def known_fraction(self, level):
        """Per-node fraction of descendant leaves that have any label at level."""
        return self.hist[level].sum(1) / self.nodes["n_leaf"].values


def categorize(idx, level, a, b, cutoff):
    """Pair-relative category per node: only_A, only_B, both, neither.

    A label is 'generated' if >cutoff of descendant leaves carry it. If b is
    None, B is the complement of A (all other labeled leaves at this level).
    """
    fa = idx.fraction(level, a)
    fb = idx.known_fraction(level) - fa if b is None else idx.fraction(level, b)
    gen_a, gen_b = fa > cutoff, fb > cutoff
    cat = np.full(len(fa), "neither", dtype=object)
    cat[gen_a & ~gen_b] = "only_A"
    cat[gen_b & ~gen_a] = "only_B"
    cat[gen_a & gen_b] = "both"
    return cat


def global_commitment(idx, level, cutoff):
    """Single dominant label per node, or 'uncommitted' if none exceeds cutoff."""
    H = idx.hist[level]
    frac = H / idx.nodes["n_leaf"].values[:, None]
    j = frac.argmax(1)
    top = frac[np.arange(len(j)), j]
    cols = np.array(idx.cols[level], dtype=object)
    out = cols[j]
    out[top <= cutoff] = "uncommitted"
    return out


def deg_cells(idx, cells_a, cells_b, name_a="A", name_b="B"):
    """Wilcoxon DEG (up = higher in group A) over two lists of cell names.

    Returns the scanpy long table: names, scores, pvals_adj, logfoldchanges.
    One row per cell; a cell repeated across nodes is repeated here too.
    """
    import anndata as ad
    import scanpy as sc

    rows = [idx.row_of[c] for c in cells_a] + [idx.row_of[c] for c in cells_b]
    grp = [name_a] * len(cells_a) + [name_b] * len(cells_b)
    adata = ad.AnnData(
        X=idx.expr[rows],
        obs=pd.DataFrame({"grp": pd.Categorical(grp)},
                         index=[str(i) for i in range(len(rows))]),
        var=pd.DataFrame(index=idx.genes),
    )
    sc.tl.rank_genes_groups(adata, "grp", groups=[name_a], reference=name_b,
                            method="wilcoxon")
    return sc.get.rank_genes_groups_df(adata, group=name_a)


def run(idx, level, a, b, cutoff=0.4, min_nodes=10, padj=0.05, n_top=3,
        max_per_group=None, seed=0):
    """Categorize nodes, then run only_A vs only_B DEGs per bin.

    max_per_group caps the nodes per group before the DEG test (random, seeded);
    a few hundred keeps the top genes stable while cutting latency. None = exact.

    Returns dict: counts (bin x category), deg (per bin: n_deg, top_a, top_b),
    and the label strings used.
    """
    rng = np.random.RandomState(seed)
    name_b = b if b is not None else f"non-{a}"
    cat = categorize(idx, level, a, b, cutoff)
    bins = idx.nodes["bin"].values
    cells = idx.nodes["cell"].values

    counts = {lab: [0] * NBIN for lab in ["only_A", "only_B", "both", "neither"]}
    for c, bi in zip(cat, bins):
        counts[c][bi] += 1

    def cap(lst):
        if max_per_group and len(lst) > max_per_group:
            return list(rng.choice(lst, max_per_group, replace=False))
        return lst

    deg = {}
    for bi in range(NBIN):
        m = bins == bi
        ca = cap([c for c in cells[m & (cat == "only_A")] if c in idx.row_of])
        cb = cap([c for c in cells[m & (cat == "only_B")] if c in idx.row_of])
        if len(ca) < min_nodes or len(cb) < min_nodes:
            deg[bi] = None
            continue
        df = deg_cells(idx, ca, cb, a, name_b)
        sig = df[df.pvals_adj < padj]
        deg[bi] = {
            "n_deg": len(sig),
            "top_a": list(sig.sort_values("scores", ascending=False).names[:n_top]),
            "top_b": list(sig.sort_values("scores").names[:n_top]),
        }
    return {"level": level, "a": a, "b": name_b, "cutoff": cutoff,
            "counts": counts, "deg": deg}


def plot(res, ax_title=None):
    """Three-panel figure: category counts, n DEGs, top DEGs per bin."""
    import matplotlib.pyplot as plt

    pal = {"only_A": "#4878a8", "only_B": "#d1615d",
           "both": "#8f6fb0", "neither": "#bab8b2"}
    x = np.arange(NBIN)
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 2, 3]})

    bottom = np.zeros(NBIN)
    for lab in ["only_A", "only_B", "both", "neither"]:
        h = np.array(res["counts"][lab])
        ax1.bar(x, h, bottom=bottom, color=pal[lab], label=lab, width=0.8)
        bottom += h
    ax1.set_ylabel("# internal nodes")
    ax1.set_title(ax_title or
                  f"{res['a']} vs {res['b']} commitment "
                  f"({res['level']}, cutoff {res['cutoff']})")
    ax1.legend(frameon=False, ncol=4, fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels([])

    ndeg = [res["deg"][bi]["n_deg"] if res["deg"][bi] else 0 for bi in range(NBIN)]
    ax2.bar(x, ndeg, color="#5b9c6b", width=0.8)
    ax2.set_ylabel("# DEGs")
    ax2.set_xticks(x); ax2.set_xticklabels(BIN_LABELS, rotation=40, ha="right",
                                           fontsize=8)
    ax2.set_xlabel("internal-node time bin")

    ax3.axis("off")
    ax3.set_title("top DEGs per bin", fontsize=10, loc="left")
    ax3.text(0.20, 1.0, "up " + res["a"], transform=ax3.transAxes,
             fontsize=8, fontweight="bold", color=pal["only_A"], va="top")
    ax3.text(0.60, 1.0, "up " + res["b"], transform=ax3.transAxes,
             fontsize=8, fontweight="bold", color=pal["only_B"], va="top")
    for r in range(NBIN):
        y = 0.9 - r * (0.9 / NBIN)
        d = res["deg"][r]
        ta = ", ".join(d["top_a"]) if d else "(too few nodes)"
        tb = ", ".join(d["top_b"]) if d else ""
        ax3.text(0.0, y, BIN_LABELS[r], transform=ax3.transAxes,
                 fontsize=8, fontweight="bold", va="top")
        ax3.text(0.20, y, ta, transform=ax3.transAxes, fontsize=8, va="top")
        ax3.text(0.60, y, tb, transform=ax3.transAxes, fontsize=8, va="top")

    for a_ in (ax1, ax2):
        a_.set_xlim(-0.6, NBIN - 0.4)
    fig.tight_layout()
    return fig
