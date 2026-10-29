"""How does each source type specify endothelial (EC) fate?

For each of the top source cell types (non-EC mesoderm/mesenchyme that gives rise
to EC), we take the internal nodes assigned to that type and split them by fate:

  EC-fated   : >EC_HI of descendant leaves are EC
  non-EC     : <EC_LO of descendant leaves are EC (a clean negative)

Per time bin we DEG the nodes' inferred states (assigned-cell expression),
EC-fated vs non-EC, giving the genes that mark a source-S cell already turning
toward EC. We then compare these signatures across sources to ask whether EC
fate is specified by the same genes everywhere (with different timing) or not.

Run from the repo root; needs the index (python -m analysis.pipelines.commitment.build_index).
"""

import os

import numpy as np
import pandas as pd

from analysis.pipelines.commitment import commitment as C

# config
EC_HI = 0.10        # EC-fated if >this fraction of descendant leaves are EC
EC_LO = 0.01        # non-EC (clean negative) if <this fraction
K_SOURCES = 5
MIN_NODES = 10      # min nodes per group to run a DEG test
MIN_SUPPORT = 30    # min source nodes in a bin to trust the EC-fated fraction
MAX_PER_GROUP = 400 # cap per group in the per-bin tests
POOL_CAP = 600      # cap per group in the pooled per-source test
PADJ = 0.05
N_TOP_BIN = 5       # top genes kept per (source, bin)
N_SIG = 25          # per-source pooled-signature size for cross-source overlap
OUT_DIR = "analysis/pipelines/commitment/results/ec_source"
SEED = 0

EC_KEYS = ("endothel", "endocard", "hematoendo")


def propagate_source(nodes, own, sources_set):
    """Per-node source identity = nearest source-typed ancestor-or-self.

    Walks up the stored nearest-ancestor pointers until it hits a node whose own
    type is one of `sources_set`; None if the lineage has no source ancestor.
    Memoised, so effectively linear.
    """
    tree = nodes["tree"].values
    node = nodes["node"].values
    anc = nodes["anc"].values
    row = {(t, n): i for i, (t, n) in enumerate(zip(tree, node))}
    memo = {}

    def resolve(start):
        chain, i = [], start
        while True:
            if i in memo:
                val = memo[i]; break
            if own[i] in sources_set:
                val = own[i]; break
            chain.append(i)
            a = anc[i]
            i = row.get((tree[i], a), -1) if a else -1
            if i < 0:
                val = None; break
        for k in chain:
            memo[k] = val
        return val

    return np.array([resolve(i) for i in range(len(nodes))], dtype=object)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.RandomState(SEED)
    idx = C.Index()

    # per-node: own assigned cell type, and EC fraction of descendant leaves
    ti = pd.read_csv(C.TYPE_INFO, index_col=0)
    sub = pd.read_csv(C.SUBTYPES, index_col=0)["cell_subtype"]
    own = idx.nodes["cell"].map(sub).map(dict(zip(ti.index, ti["cell_type"]))).values

    cols = idx.cols["cell_type"]
    ec_cols = [c for c in cols if any(k in c.lower() for k in EC_KEYS)]
    ec_idx = [cols.index(c) for c in ec_cols]
    n_leaf = idx.nodes["n_leaf"].values
    ec_frac = idx.hist["cell_type"][:, ec_idx].sum(1) / n_leaf
    bins = idx.nodes["bin"].values
    cells = idx.nodes["cell"].values
    is_ec_type = pd.Series(own).isin(ec_cols).values

    # EC-fated / clean-negative by descendant leaves (EC-typed nodes included, so
    # the EC descendants of a source are retained after the label switch)
    fated = ec_frac > EC_HI
    neg = ec_frac < EC_LO

    # sources ranked by their own-typed EC-fated nodes (which types originate EC)
    sources = pd.Series(own[(~is_ec_type) & fated]).value_counts(
        ).head(K_SOURCES).index.tolist()
    print("top sources (by # EC-fated nodes):", sources)

    # admissible set per source = source-typed nodes AND their descendants: each
    # node inherits the identity of its nearest source-typed ancestor-or-self
    src = propagate_source(idx.nodes, own, set(sources))

    def cap(names):
        names = [c for c in names if c in idx.row_of]
        if len(names) > MAX_PER_GROUP:
            names = list(rng.choice(names, MAX_PER_GROUP, replace=False))
        return names

    # per source: counts + DEG per bin
    summary = {}      # source -> dict(n_fated, n_neg, frac, ndeg, top_up per bin)
    combos = {}       # source -> DataFrame(gene, bin, score) for significant up genes
    for s in sources:
        s_mask = (src == s)
        n_fated, n_neg, frac, ndeg = ([0] * C.NBIN for _ in range(4))
        top_up = [None] * C.NBIN
        rows = []
        for b in range(C.NBIN):
            in_bin = s_mask & (bins == b)
            f_cells = list(cells[in_bin & fated])
            g_cells = list(cells[in_bin & neg])
            n_fated[b], n_neg[b] = len(f_cells), len(g_cells)
            tot = len(f_cells) + len(g_cells)
            frac[b] = len(f_cells) / tot if tot else np.nan
            ca, cb = cap(f_cells), cap(g_cells)
            if len(ca) < MIN_NODES or len(cb) < MIN_NODES:
                continue
            df = C.deg_cells(idx, ca, cb, "EC", "nonEC")
            sig = df[df.pvals_adj < PADJ]
            up = sig[sig.scores > 0].sort_values("scores", ascending=False)
            ndeg[b] = len(sig)
            top_up[b] = list(up.names[:N_TOP_BIN])
            for _, r in up.head(50).iterrows():   # keep a deep list for combo ranking
                rows.append((r["names"], b, r["scores"]))
        # mask the fraction where too few source nodes support it
        support = np.array(n_fated) + np.array(n_neg)
        frac = [f if support[b] >= MIN_SUPPORT else np.nan for b, f in enumerate(frac)]
        summary[s] = dict(n_fated=n_fated, n_neg=n_neg, frac=frac, ndeg=ndeg,
                          top_up=top_up, pooled_sig=pooled_signature(
                              idx, cap, s_mask, fated, neg, cells))
        combos[s] = pd.DataFrame(rows, columns=["gene", "bin", "score"])

    report(idx, sources, summary, combos)


def pooled_signature(idx, cap, s_mask, fated, neg, cells):
    """Robust per-source EC signature: one DEG over all bins pooled.

    Returns the significant up-in-EC genes ordered by score (or [] if too few).
    """
    ca = cap(list(cells[s_mask & fated]))[:POOL_CAP]
    cb = cap(list(cells[s_mask & neg]))[:POOL_CAP]
    if len(ca) < MIN_NODES or len(cb) < MIN_NODES:
        return []
    df = C.deg_cells(idx, ca, cb, "EC", "nonEC")
    up = df[(df.pvals_adj < PADJ) & (df.scores > 0)].sort_values(
        "scores", ascending=False)
    return list(up.names)


def report(idx, sources, summary, combos):
    # (1) top 10 (gene, time) DEG combinations per source
    print("\n=== top 10 (gene, time-bin) EC-specification signals per source ===")
    top_combos = {}
    for s in sources:
        d = combos[s].sort_values("score", ascending=False).head(10)
        top_combos[s] = d
        print(f"\n{s}:")
        for _, r in d.iterrows():
            print(f"  {r.gene:12s} @ {C.BIN_LABELS[int(r.bin)]}  score={r.score:6.1f}")
        d.assign(bin_label=[C.BIN_LABELS[int(b)] for b in d.bin]).to_csv(
            os.path.join(OUT_DIR, f"top_combos_{s.replace(' ', '_')}.csv"), index=False)

    # (2) cross-source overlap from the robust pooled per-source signatures
    sig_genes = {s: summary[s]["pooled_sig"][:N_SIG] for s in sources}
    J = np.zeros((len(sources), len(sources)))
    for i, si in enumerate(sources):
        for j, sj in enumerate(sources):
            a, b = set(sig_genes[si]), set(sig_genes[sj])
            J[i, j] = len(a & b) / len(a | b) if a | b else 0
    from collections import Counter
    shared = Counter(g for s in sources for g in sig_genes[s])
    core = [g for g, c in shared.most_common() if c >= max(2, len(sources) - 1)]
    print(f"\n=== cross-source pooled-signature overlap (top {N_SIG} genes each) ===")
    print("genes shared by >= %d/%d sources:" % (max(2, len(sources) - 1), len(sources)))
    for g, c in shared.most_common(15):
        print(f"  {g:12s} in {c}/{len(sources)} sources")

    # (3) timing per source: peak by absolute EC-fated node count (robust) + #DEG
    print("\n=== timing per source ===")
    peak = {}
    for s in sources:
        nf = np.array(summary[s]["n_fated"], dtype=float)
        nd = np.array(summary[s]["ndeg"], dtype=float)
        pf = int(nf.argmax()) if nf.any() else -1
        pd_ = int(nd.argmax()) if nd.any() else -1
        peak[s] = (pf, pd_)
        print(f"  {s:26s} peak EC-fated count @ {C.BIN_LABELS[pf]:11s} "
              f"peak #DEG @ {C.BIN_LABELS[pd_]}")

    conceptual_takeaway(sources, sig_genes, core, peak)
    make_figure(sources, summary, J)


def conceptual_takeaway(sources, sig_genes, core, peak):
    n = len(sources)
    from collections import Counter
    shared = Counter(g for s in sources for g in sig_genes[s])
    frac_shared = np.mean([shared[g] >= n - 1 for s in sources for g in sig_genes[s]])
    peak_bins = sorted({peak[s][0] for s in sources})   # peak EC-fated-count bin
    spread = (max(peak_bins) - min(peak_bins)) if peak_bins else 0
    print("\n=== conceptual takeaway (data-driven) ===")
    print(f"* {len(core)} genes recur across >= {n-1}/{n} sources; "
          f"{100*frac_shared:.0f}% of per-source top-{len(sig_genes[sources[0]])} "
          f"genes are broadly shared.")
    print(f"* peak EC-fated production spans {spread} time bins across sources "
          f"({', '.join(C.BIN_LABELS[b] for b in peak_bins)}).")
    verdict = ("shared core program, staggered timing" if core and spread >= 1
               else "shared core program, similar timing" if core
               else "largely source-specific programs")
    print(f"* reading: {verdict}. A shared core ({', '.join(core[:6]) or 'n/a'}) "
          f"points to a common EC-specification module; each source also carries "
          f"private genes and engages the program on its own schedule.")


def make_figure(sources, summary, J):
    import matplotlib.pyplot as plt

    colors = ["#4878a8", "#d1615d", "#5b9c6b", "#b07aa1", "#e0913a"]
    x = np.arange(C.NBIN)
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))

    for s, c in zip(sources, colors):
        ax[0].plot(x, summary[s]["n_fated"], "-o", color=c, ms=3, label=s)
        ax[1].plot(x, summary[s]["frac"], "-o", color=c, ms=3, label=s)
        ax[2].plot(x, summary[s]["ndeg"], "-o", color=c, ms=3, label=s)
    ax[0].set_title("# EC-fated nodes over time"); ax[0].set_ylabel("# EC-fated source nodes")
    ax[1].set_title("fraction EC-fated over time"); ax[1].set_ylabel("EC-fated / source nodes")
    ax[2].set_title("# EC-specification DEGs over time"); ax[2].set_ylabel("# DEGs")
    for a in ax[:3]:
        a.set_xticks(x); a.set_xticklabels(C.BIN_LABELS, rotation=40, ha="right", fontsize=7)
        a.set_xlabel("node time bin")
    ax[0].legend(fontsize=6, frameon=False)

    im = ax[3].imshow(J, cmap="magma", vmin=0, vmax=1)
    ax[3].set_xticks(range(len(sources))); ax[3].set_yticks(range(len(sources)))
    short = [s.replace(" mesoderm", " meso") for s in sources]
    ax[3].set_xticklabels(short, rotation=40, ha="right", fontsize=7)
    ax[3].set_yticklabels(short, fontsize=7)
    ax[3].set_title(f"pooled signature overlap (Jaccard, top {N_SIG})")
    fig.colorbar(im, ax=ax[3], shrink=0.7)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "ec_source_specification.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
