"""row-2 cache (Hox-code consistency), computed from the assignment pkls.

Pick a transcriptionally-defined cell type, take all cells of that type observed
at E9.5 (the tree leaves), and walk each leaf back up its lineage to the ancestor
node sitting at ~E9.0. That ancestor is *unobserved*, so each method imputes it by
assigning it to an observed cell (the assignment pkl). For a Hox gene believed to be
statically expressed in that cell type, a faithful assignment should hand the
ancestor a cell whose Hox level matches the descendant's -> high ancestor/descendant
correlation. A method that scrambles lineage will assign Hox-mismatched ancestors.

We score the (cell type, Hox gene) shortlist in literature/hox_consistency.py on the
COMMON SUPPORT -- (leaf, ancestor) pairs where ALL THREE methods assign the ancestor
-- so every method is scored on an identical set of cells. 
Per (method, cell type, gene) we report the ancestor/descendant Pearson r and a
"distortion" = 1 - r, so that lower is better.
"""
import os
import sys
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from analysis.figures.summary_qc import style as S
from datasets.mouse_embryo import MouseEmbryoDataset
from literature.hox_consistency import static_hox_pairs

LEAF_TIME = 9.5          # cell type is defined transcriptionally at the leaves (E9.5)
ANCESTOR_TIME = 9.0      # walk each leaf back to its ancestor at ~E9.0
ANCESTOR_TOL = 0.4       # accept the nearest ancestor within this of E9.0
TIME_ATOL = 1e-3         # tolerance for matching a node's stored time to LEAF_TIME


def load_assignments():
    """{method: {tree_name: {node: observed_cell}}} for the three methods."""
    out = {}
    with open(S.MODEL_ASSIGNMENTS, "rb") as f:
        out["model"] = pickle.load(f)          # carries an extra 'nn_distances' key; harmless
    for method, pkl in S.GENOT_PKLS.items():
        with open(pkl, "rb") as f:
            out[method] = pickle.load(f)["assignments"]
    return out


def find_ancestor_near(tree, leaf, target_time, tol):
    """The strict ancestor of ``leaf`` whose time is nearest ``target_time``.

    Walks parent links from ``leaf`` toward the root, returning the internal
    node minimising ``|time - target_time|``; returns None if the nearest such
    ancestor is further than ``tol`` from ``target_time``.
    """
    best, best_d = None, float("inf")
    preds = list(tree.predecessors(leaf))
    cur = preds[0] if preds else None
    while cur is not None:
        t = tree.nodes[cur].get("time")
        if t is not None:
            d = abs(t - target_time)
            if d < best_d:
                best, best_d = cur, d
        preds = list(tree.predecessors(cur))
        cur = preds[0] if preds else None
    return best if best_d <= tol else None


def collect_pairs(dataset, assignments, relevant_cts):
    """Common-support (leaf, ancestor_cell) pairs per (method, cell type).

    Walks the trees ONCE (the leaf set + the ~E9.0 ancestor node are
    method-independent), then looks up the cell each method assigns that ancestor
    to, keeping only ancestors ALL methods assign. ``coverage`` records, per
    method, how many ancestors it could assign before the intersection.
    """
    name_to_ct = dataset.states.obs["cell_type"].to_dict()
    pairs = defaultdict(list)          # (method, ct) -> [(leaf, anc_cell), ...]
    n_leaves = defaultdict(int)        # ct -> E9.5 leaves of that type seen
    n_anc_found = defaultdict(int)     # ct -> leaves with a ~E9.0 ancestor
    n_common = defaultdict(int)        # ct -> common-support pairs kept
    coverage = defaultdict(int)        # method -> ancestors it could assign
    for tname, tree in dataset.trees.items():
        for leaf in tree.nodes():
            if tree.out_degree(leaf) != 0:
                continue
            if not np.isclose(tree.nodes[leaf].get("time", -1), LEAF_TIME, atol=TIME_ATOL):
                continue
            ct = name_to_ct.get(leaf)
            if ct not in relevant_cts:
                continue
            n_leaves[ct] += 1
            anc = find_ancestor_near(tree, leaf, ANCESTOR_TIME, ANCESTOR_TOL)
            if anc is None:
                continue
            n_anc_found[ct] += 1
            anc_cells = {}
            for method in S.METHODS:
                cell = assignments[method].get(tname, {}).get(anc)
                if cell is not None:
                    coverage[method] += 1
                    anc_cells[method] = cell
            if len(anc_cells) != len(S.METHODS):     # require common support
                continue
            n_common[ct] += 1
            for method in S.METHODS:
                pairs[(method, ct)].append((leaf, anc_cells[method]))
    return pairs, n_leaves, n_anc_found, n_common, coverage


def main():
    print("loading dataset...", flush=True)
    dataset = MouseEmbryoDataset(max_size=S.N_TREES)

    # group the shortlist: {cell_type: [hox genes]}, keeping only HVG-present genes
    genes_by_ct = defaultdict(list)
    for ct, gene in static_hox_pairs:
        genes_by_ct[ct].append(gene)
    wanted_genes = sorted({g for _, g in static_hox_pairs})
    present = [g for g in wanted_genes if g in set(dataset.states.var_names)]
    missing = [g for g in wanted_genes if g not in set(dataset.states.var_names)]
    if missing:
        print(f"NOTE: {len(missing)} shortlisted Hox genes absent from the HVG set, "
              f"skipped: {', '.join(missing)}", flush=True)

    # dense (n_cells, n_present) Hox expression, row-indexed by name_to_idx
    print(f"pulling expression for {len(present)} Hox genes...", flush=True)
    hox_expr = np.asarray(dataset.states[:, present].X.todense(), dtype=np.float32)
    gene_col = {g: j for j, g in enumerate(present)}
    name_to_idx = dataset.name_to_idx

    def expr(cell, gene):
        idx = name_to_idx.get(cell)
        return hox_expr[idx, gene_col[gene]] if idx is not None else None

    assignments = load_assignments()
    relevant_cts = set(genes_by_ct)

    print("collecting leaf->ancestor pairs...", flush=True)
    pairs, n_leaves, n_anc_found, n_common, coverage = collect_pairs(
        dataset, assignments, relevant_cts)

    print("\nper-cell-type counts (E9.5 leaves / with ~E9.0 ancestor / common-support):")
    print(f"  {'cell type':<28s}{'leaves':>9s}{'w/anc':>9s}{'common':>9s}")
    for ct in sorted(relevant_cts):
        print(f"  {ct:<28s}{n_leaves[ct]:>9d}{n_anc_found[ct]:>9d}{n_common[ct]:>9d}")

    total_anc = sum(n_anc_found.values())
    print("\nancestor-assignment coverage (before intersection), of "
          f"{total_anc} leaves-with-ancestor:")
    for method in S.METHODS:
        c = coverage[method]
        print(f"  {S.METHOD_LABELS[method]:<20s}{c:>8d}  ({100.0 * c / total_anc:.2f}%)")

    # correlate ancestor vs descendant Hox expression, per method x ct x gene;
    # one CSV row per scorable (method, cell type, gene), distortion = 1 - r.
    rows = []
    for method in S.METHODS:
        for ct in sorted(genes_by_ct):
            plist = pairs[(method, ct)]
            for gene in genes_by_ct[ct]:
                if gene not in gene_col:
                    continue
                desc, anc = [], []
                for leaf, anc_cell in plist:
                    dv, av = expr(leaf, gene), expr(anc_cell, gene)
                    if dv is None or av is None:
                        continue
                    desc.append(dv)
                    anc.append(av)
                n = len(desc)
                if n >= 3 and np.std(desc) > 0 and np.std(anc) > 0:
                    r, _ = pearsonr(desc, anc)
                    rows.append((method, ct, gene, n, r, 1.0 - r))

    df = pd.DataFrame(rows, columns=["method", "cell_type", "gene", "n", "r", "distortion"])
    out = os.path.join(S.NR, "hox_consistency.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  ({len(df)} rows)", flush=True)

    print("\n=== per-method summary (mean over ct x gene) ===")
    print(f"{'method':<20s}{'mean r':>9s}{'mean distortion':>18s}{'n pairs':>10s}")
    for method in S.METHODS:
        sub = df[df["method"] == method]
        print(f"{S.METHOD_LABELS[method]:<20s}{sub['r'].mean():>9.3f}"
              f"{sub['distortion'].mean():>18.3f}{len(sub):>10d}")


if __name__ == "__main__":
    main()
