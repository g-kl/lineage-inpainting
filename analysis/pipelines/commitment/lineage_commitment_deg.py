#!/usr/bin/env python
"""When do internal nodes commit to one lineage, and how do committed states differ?

A "fate commitment" analysis on the E9.5 (9.5-leaf-time) trees. Given a
pair of leaf labels A and B (here the lineages neural ectoderm vs neural crest):

  1. label every leaf A / B / other (via cell_subtype -> lineage);
  2. for each internal node, count how many of its descendant leaves are A vs B
     (one post-order pass per tree), and categorise the node by which labels it
     "generates" -- a label is generated if >CUTOFF of its descendant leaves
     carry it. Four categories: only-A, only-B, both, neither;
  3. bin the internal-node times into (7.50,7.75) ... (9.25,9.50) quarter-Myr bins;
  4. per bin, if at least MIN_NODES nodes generate *only one* label, run a
     Wilcoxon DEG test (only-A vs only-B) on the nodes' inferred states -- the
     expression of the observed cell each node is nn-coupled to (assignments pkl).

Only nodes that have an inferred state (an assignment) can enter categorisation
and DEGs, so the counts below are over assigned internal nodes.

Outputs a 3-panel figure (category counts / #DEGs / top-3 DEGs per bin) and a
per-phase runtime summary. Run from the repo root (so ../data/embryos resolves).
"""

import os
import pickle
import time
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DATA_DIR = "../data/embryos"
H5AD = os.path.join(DATA_DIR, "qiu24_integrated_hvgs_and_scvi.h5ad")
TREES = os.path.join(DATA_DIR, "final_topology.h5td")
TYPE_INFO = os.path.join(DATA_DIR, "final_type_info_v2_lower.csv")
SUBTYPES = os.path.join(DATA_DIR, "final_subtype_v2_lower.csv")
ASSIGN_PKL = (
    "outputs/mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none"
    "_57ab441ec3fcecf81d545755db5ddb09/"
    "assignments_nn_mouse_embryo_grid_bridge_unshared_vpsde_idnn_leaf_nosa_std_none"
    "_57ab441ec3fcecf81d545755db5ddb09_step20000_adj_n1.pkl"
)

# The label pair. LABEL_LEVEL is a column of type_info ("lineage" / "cell_type").
# Set LABEL_B = None for a complement analysis ("A vs not-A"): every non-A leaf
# with a known label counts as B.
LABEL_LEVEL = "lineage"
LABEL_A = "neural ectoderm"
LABEL_B = "neural crest"

CUTOFF = 0.4        # a label is "generated" if >CUTOFF of descendant leaves carry it
MIN_NODES = 10      # min only-one-label nodes per group to attempt a DEG test
BIN_EDGES = np.round(np.arange(7.5, 9.5 + 1e-9, 0.25), 2)   # (7.50,7.75)..(9.25,9.50)
EPS = 1e-6          # float-error slack for time binning
LEAF_TIME = 9.5     # only trees whose leaves sit at this time
PADJ = 0.05         # DEG significance cutoff (BH-adjusted)
N_TOP = 3           # top DEGs per direction to report/plot

OUT_DIR = "analysis/pipelines/commitment/results"
UNKNOWN = "unknown"

# category codes for leaves and node buckets
A, B, OTHER = 0, 1, 2
ONLY_A, ONLY_B, BOTH, NEITHER = "only_A", "only_B", "both", "neither"
CATEGORIES = [ONLY_A, ONLY_B, BOTH, NEITHER]


class Timer:
    """Accumulates wall-clock time per named phase for the runtime summary."""

    def __init__(self):
        self.t = {}

    def __call__(self, name):
        self._name, self._t0 = name, time.perf_counter()
        return self

    def __enter__(self):
        print(f"[{self._name}] ...")
        return self

    def __exit__(self, *exc):
        self.t[self._name] = self.t.get(self._name, 0.0) + time.perf_counter() - self._t0


timer = Timer()


# --------------------------------------------------------------------------- #
# leaf labels: cell -> {A, B, OTHER}
# --------------------------------------------------------------------------- #
with timer("load labels"):
    ti = pd.read_csv(TYPE_INFO, index_col=0)
    sub2label = dict(zip(ti.index, ti[LABEL_LEVEL]))       # cell_subtype -> lineage/type
    subtype = pd.read_csv(SUBTYPES, index_col=0)["cell_subtype"]

    label = subtype.map(sub2label)                          # cell -> label string
    code = np.full(len(label), OTHER, dtype=np.int8)
    code[label.values == LABEL_A] = A
    if LABEL_B is None:                                     # complement: any other known label is B
        code[(label.values != LABEL_A) & label.notna().values] = B
    else:
        code[label.values == LABEL_B] = B
    cell2code = dict(zip(subtype.index, code))              # O(1) leaf lookup during traversal

name_B = LABEL_B if LABEL_B is not None else f"non-{LABEL_A}"
print(f"      A = {LABEL_A!r}, B = {name_B!r} ({LABEL_LEVEL}); cutoff={CUTOFF}")


# --------------------------------------------------------------------------- #
# trees (9.5-leaf-time only) + assignments (node -> observed cell)
# --------------------------------------------------------------------------- #
with timer("load trees"):
    import treedata

    td = treedata.read_h5td(TREES)
    tree_names = [n for n in td.obst if n.startswith(f"E{LEAF_TIME}")]
    print(f"      {len(tree_names)} trees at leaf time {LEAF_TIME}: {tree_names}")

with timer("load assignments"):
    with open(ASSIGN_PKL, "rb") as fh:
        assignments = pickle.load(fh)


# --------------------------------------------------------------------------- #
# per tree: descendant A/B/leaf counts (one post-order pass) -> categorise the
# assigned internal nodes and record (time_bin, category, cell)
# --------------------------------------------------------------------------- #
def bin_of(t):
    """Index of the quarter-Myr bin containing time t, or -1 if outside the range."""
    if t < BIN_EDGES[0] - EPS or t >= BIN_EDGES[-1] - EPS:
        return -1
    return int(np.floor((t - BIN_EDGES[0]) / 0.25 + EPS))


def descendant_counts(g):
    """Return dicts nA, nB, nL: descendant A / B / total-leaf counts per node.

    Single reverse-topological (post-order) sweep: a leaf seeds its own code,
    an internal node sums its children. O(nodes)."""
    nA, nB, nL = {}, {}, {}
    for node in reversed(list(__import__("networkx").topological_sort(g))):
        succ = list(g.successors(node))
        if not succ:                                        # leaf
            c = cell2code.get(node, OTHER)
            nA[node], nB[node], nL[node] = int(c == A), int(c == B), 1
        else:
            nA[node] = sum(nA[s] for s in succ)
            nB[node] = sum(nB[s] for s in succ)
            nL[node] = sum(nL[s] for s in succ)
    return nA, nB, nL


def categorise(fa, fb):
    gen_a, gen_b = fa > CUTOFF, fb > CUTOFF
    if gen_a and gen_b:
        return BOTH
    if gen_a:
        return ONLY_A
    if gen_b:
        return ONLY_B
    return NEITHER


with timer("categorise nodes"):
    # bin_cat_count[bin][category] -> n nodes; committed[bin][cat] -> [cell, ...]
    bin_cat_count = defaultdict(lambda: defaultdict(int))
    committed = defaultdict(lambda: {ONLY_A: [], ONLY_B: []})

    for tname in tree_names:
        g = td.obst[tname]
        amap = assignments.get(tname, {})
        nA, nB, nL = descendant_counts(g)
        for node, cell in amap.items():
            if cell == UNKNOWN or g.out_degree(node) == 0:  # skip unknowns and leaves
                continue
            bi = bin_of(float(g.nodes[node].get("time", np.nan)))
            if bi < 0:
                continue
            tot = nL[node]
            cat = categorise(nA[node] / tot, nB[node] / tot)
            bin_cat_count[bi][cat] += 1
            if cat in (ONLY_A, ONLY_B):
                committed[bi][cat].append(cell)

    n_assigned = sum(sum(c.values()) for c in bin_cat_count.values())
    print(f"      categorised {n_assigned:,} assigned internal nodes across "
          f"{len(bin_cat_count)} bins")


# --------------------------------------------------------------------------- #
# read inferred states (expression of each node's assigned cell) straight from
# h5py -- one CSR row per cell -- for the committed nodes only
# --------------------------------------------------------------------------- #
with timer("read expression"):
    need = sorted({c for bi in committed for cat in (ONLY_A, ONLY_B)
                   for c in committed[bi][cat]})

    with h5py.File(H5AD, "r") as f:
        obs_names = np.array([n.decode() for n in f["obs"]["_index"][:]])
        genes = np.array([g.decode() for g in f["var"]["_index"][:]])
    name2row = {n: i for i, n in enumerate(obs_names)}
    rows = sorted({name2row[c] for c in need if c in name2row})

    def read_rows(rows):
        with h5py.File(H5AD, "r") as f:
            X = f["X"]
            data, indices, indptr = X["data"], X["indices"], X["indptr"][:]
            ncol = int(X.attrs["shape"][1])
            d, i, ptr = [], [], [0]
            for r in rows:                                  # gather CSR rows one by one
                s, e = int(indptr[r]), int(indptr[r + 1])
                d.append(data[s:e]); i.append(indices[s:e]); ptr.append(ptr[-1] + (e - s))
        return sp.csr_matrix((np.concatenate(d), np.concatenate(i), np.array(ptr)),
                             shape=(len(rows), ncol))

    Xmat = read_rows(rows)                                  # (cells, genes), lognorm
    cell2local = {obs_names[r]: i for i, r in enumerate(rows)}
    print(f"      {Xmat.shape[0]:,} unique committed-node cells x {Xmat.shape[1]:,} genes")


# --------------------------------------------------------------------------- #
# per bin: only-A vs only-B Wilcoxon DEG (one row per node = its inferred state)
# --------------------------------------------------------------------------- #
with timer("DEG tests"):
    import anndata as ad
    import scanpy as sc

    deg_summary = {}        # bin -> {"n_deg", "top_A":[...], "top_B":[...]} or None
    for bi in range(len(BIN_EDGES) - 1):
        cells_a = [c for c in committed[bi][ONLY_A] if c in cell2local]
        cells_b = [c for c in committed[bi][ONLY_B] if c in cell2local]
        if len(cells_a) < MIN_NODES or len(cells_b) < MIN_NODES:
            deg_summary[bi] = None
            continue

        cells = cells_a + cells_b
        groups = [LABEL_A] * len(cells_a) + [name_B] * len(cells_b)
        adata = ad.AnnData(
            X=Xmat[[cell2local[c] for c in cells]],
            obs=pd.DataFrame({"grp": pd.Categorical(groups)},
                             index=[f"{c}#{k}" for k, c in enumerate(cells)]),  # nodes may share a cell
            var=pd.DataFrame(index=genes),
        )
        sc.tl.rank_genes_groups(adata, "grp", groups=[LABEL_A], reference=name_B,
                                method="wilcoxon")
        df = sc.get.rank_genes_groups_df(adata, group=LABEL_A)   # +score = up in A
        n_deg = int((df.pvals_adj < PADJ).sum())
        sig = df[df.pvals_adj < PADJ]
        top_A = list(sig.sort_values("scores", ascending=False).names[:N_TOP])
        top_B = list(sig.sort_values("scores").names[:N_TOP])
        deg_summary[bi] = {"n_deg": n_deg, "top_A": top_A, "top_B": top_B}


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
labels = [f"{BIN_EDGES[i]:.2f}-{BIN_EDGES[i + 1]:.2f}" for i in range(len(BIN_EDGES) - 1)]
print("\n=== per-bin summary ===")
for bi in range(len(BIN_EDGES) - 1):
    cc = bin_cat_count[bi]
    parts = "  ".join(f"{k}={cc.get(k, 0)}" for k in CATEGORIES)
    print(f"[{labels[bi]}] {parts}")
    d = deg_summary.get(bi)
    if d is None:
        print(f"           DEGs: (skipped; need >={MIN_NODES} only-A and only-B nodes)")
    else:
        print(f"           DEGs: {d['n_deg']} sig | up-{LABEL_A}: {d['top_A']} | "
              f"up-{name_B}: {d['top_B']}")


# --------------------------------------------------------------------------- #
# plot: (1) category counts, (2) #DEGs, (3) top-3 DEG table, per bin
# --------------------------------------------------------------------------- #
with timer("plot"):
    import matplotlib.pyplot as plt

    PAL = {ONLY_A: "#4878a8", ONLY_B: "#d1615d", BOTH: "#8f6fb0", NEITHER: "#bab8b2"}
    x = np.arange(len(labels))
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(8, 9), gridspec_kw={"height_ratios": [3, 2, 3]})

    # (1) stacked category counts
    bottom = np.zeros(len(labels))
    for cat in CATEGORIES:
        h = np.array([bin_cat_count[bi].get(cat, 0) for bi in range(len(labels))])
        ax1.bar(x, h, bottom=bottom, color=PAL[cat], label=cat, width=0.8)
        bottom += h
    ax1.set_ylabel("# internal nodes")
    ax1.set_title(f"{LABEL_A} vs {name_B} commitment across time "
                  f"({LABEL_LEVEL}, cutoff {CUTOFF})")
    ax1.legend(frameon=False, ncol=4, fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels([])

    # (2) number of significant DEGs
    ndeg = [deg_summary[bi]["n_deg"] if deg_summary.get(bi) else 0
            for bi in range(len(labels))]
    ax2.bar(x, ndeg, color="#5b9c6b", width=0.8)
    ax2.set_ylabel(f"# DEGs (padj<{PADJ})")
    ax2.set_xticks(x); ax2.set_xticklabels([])

    # (3) top-3 DEG table -- one row per bin, columns: bin | up-A | up-B
    ax3.axis("off")
    ax3.set_title("top 3 DEGs per bin", fontsize=10, loc="left")
    col_x = [0.0, 0.20, 0.60]                              # bin / up-A / up-B columns
    ax3.text(col_x[1], 1.0, f"↑ {LABEL_A}", transform=ax3.transAxes,
             fontsize=8, fontweight="bold", va="top", color=PAL[ONLY_A])
    ax3.text(col_x[2], 1.0, f"↑ {name_B}", transform=ax3.transAxes,
             fontsize=8, fontweight="bold", va="top", color=PAL[ONLY_B])
    for r, bi in enumerate(range(len(labels))):
        y = 0.9 - r * (0.9 / len(labels))                 # evenly stacked rows
        d = deg_summary.get(bi)
        up_a = ", ".join(d["top_A"]) if d else "(too few nodes)"
        up_b = ", ".join(d["top_B"]) if d else ""
        ax3.text(col_x[0], y, labels[bi], transform=ax3.transAxes,
                 fontsize=8, fontweight="bold", va="top")
        ax3.text(col_x[1], y, up_a, transform=ax3.transAxes, fontsize=8, va="top")
        ax3.text(col_x[2], y, up_b, transform=ax3.transAxes, fontsize=8, va="top")

    ax2.set_xlabel("internal-node time bin")
    for bi_ax in (ax1, ax2):
        bi_ax.set_xlim(-0.6, len(labels) - 0.4)
    ax2.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax2.xaxis.set_tick_params(labelbottom=True)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{LABEL_A}_vs_{name_B}".replace(" ", "-")
    out = os.path.join(OUT_DIR, f"commitment_deg_{tag}.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")


# --------------------------------------------------------------------------- #
# runtime summary
# --------------------------------------------------------------------------- #
print("\n=== runtime (s) ===")
total = sum(timer.t.values())
for name, dt in sorted(timer.t.items(), key=lambda kv: -kv[1]):
    print(f"  {dt:7.2f}  {100 * dt / total:5.1f}%  {name}")
print(f"  {total:7.2f}  100.0%  TOTAL")
