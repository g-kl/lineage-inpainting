"""fan-in caches: where a chosen E9.5 fate comes from.

for each target in style.FANIN, bin its E9.5 fan-in by the target's hierarchy
(lineage or cell_type) and cache per target key:
  - {key}_composition.csv  source-group fractions (kept groups + "other"), counting
                           progenitor NODES in the style.BIN_LO/BIN_HI window with a
                           fate-family descendant leaf
  - {key}_leaves.pkl       leaf + ancestor UMAP coords, coloured by source group
  - {key}_subtrees.pkl     example subtrees each dominated by one source group
the composition counts progenitor nodes (robust to when a source commits); the UMAP
links and subtrees trace focus leaves back to their ~t_anc ancestor (per fate).

reads only the tree topology + coupling + annotation CSVs.

    python analysis/figures/convergence/compute_fanin.py [key ...]
"""
import os
import pickle
import sys

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import style as S

DISPLAY_LINES = 1500     # ancestor->descendant links drawn on the UMAP


def _descendant_frac(g, is_target, is_labeled):
    """per node: (# target-type, # labeled) descendant leaves, via a leaves-up pass."""
    tgt, tot = {}, {}
    for n in reversed(list(nx.topological_sort(g))):
        succ = list(g.successors(n))
        if not succ:
            tot[n] = 1 if is_labeled(n) else 0
            tgt[n] = 1 if is_target(n) else 0
        else:
            tot[n] = sum(tot[c] for c in succ)
            tgt[n] = sum(tgt[c] for c in succ)
    return tgt, tot


def _source_types(td, e95, assignments, subtype, sub2type, family):
    """the source cell type of every internal E9.5 node in the style.BIN_LO/BIN_HI
    window that has a fate-family descendant leaf. nodes already in the family are
    dropped, so only the fan-in from other tissues is counted."""
    tgt_subtypes = {s for s, ct in sub2type.items() if ct in family}
    is_tgt = lambda n: subtype.get(n) in tgt_subtypes
    is_lab = lambda n: subtype.get(n) is not None
    out = []
    for tname in e95:
        g = td.obst[tname]
        amap = assignments.get(tname, {})
        tgt_c, tot_c = _descendant_frac(g, is_tgt, is_lab)
        for n in g.nodes():
            if g.out_degree(n) == 0 or tgt_c[n] == 0:
                continue
            if not (S.BIN_LO <= float(g.nodes[n].get("time", np.nan)) < S.BIN_HI):
                continue
            cell = amap.get(n)
            if not cell or cell == "unknown":
                continue
            st = sub2type.get(subtype.get(cell))
            if st is not None and st not in family:
                out.append(st)
    return np.array(out)


def _nearest_ancestor(g, leaf, T, tol):
    """the strict ancestor whose node time is closest to T (within tol), or None."""
    chain, cur = [], leaf
    while True:
        preds = list(g.predecessors(cur))
        if not preds:
            break
        cur = preds[0]
        chain.append(cur)
    if not chain:
        return None
    ts = np.array([float(g.nodes[n].get("time", np.nan)) for n in chain])
    j = int(np.nanargmin(np.abs(ts - T)))
    return chain[j] if abs(ts[j] - T) <= tol else None


def _square_layout(g, root):
    """rectangular dendrogram: x = node time, y = leaf order, internal node at the
    mean y of its direct children."""
    pos, counter = {}, [0]

    def assign(n):
        kids = list(g.successors(n))
        x = float(g.nodes[n].get("time", np.nan))
        if not kids:
            y = counter[0]; counter[0] += 1
            pos[n] = (x, y)
            return y
        y = float(np.mean([assign(c) for c in kids]))
        pos[n] = (x, y)
        return y

    assign(root)
    return pos


def _oscillates(g, root, leaves, group_at):
    """True if any root->leaf path revisits a group after leaving it (e.g. A,B,A) --
    a flip-flop that signals a noisy reconstruction. unknown (None) nodes are
    skipped; consecutive repeats are collapsed before checking for a re-visit."""
    for leaf in leaves:
        seq, cur = [], leaf
        while True:
            seq.append(cur)
            if cur == root:
                break
            cur = next(iter(g.predecessors(cur)))
        gs = [group_at(n) for n in reversed(seq)]
        gs = [x for x in gs if x is not None]
        runs = [x for i, x in enumerate(gs) if i == 0 or x != gs[i - 1]]
        if len(set(runs)) < len(runs):
            return True
    return False


def _trace(td, assignments, cfg, celltype_of, group_of):
    """every E9.5 focus leaf + its ~T_ANC ancestor, binned into a source group."""
    recs = []
    for tname in [n for n in td.obst if n.startswith("E9.5")]:
        g = td.obst[tname]
        amap = assignments.get(tname, {})
        for leaf in g.nodes():
            if g.out_degree(leaf) > 0:
                continue
            lt = float(g.nodes[leaf].get("time", np.nan))
            if np.isnan(lt) or abs(lt - S.ROW2_LEAF_TIME) > S.CONV_LEAF_TIME_TOL:
                continue
            if celltype_of(leaf) != cfg["focus"]:
                continue
            anc = _nearest_ancestor(g, leaf, cfg["t_anc"], S.ROW2_ANC_TOL)
            if anc is None:
                continue
            acell = amap.get(anc)
            if not acell or acell == "unknown":
                continue
            recs.append((tname, leaf, anc, acell, celltype_of(acell)))
    df = pd.DataFrame(recs, columns=["tree", "cell", "anc_node", "anc_cell",
                                     "anc_type"])
    df = df[~df["anc_type"].isin(cfg["exclude"])]
    df["src_group"] = df["anc_type"].map(group_of)
    return df   # unmapped groups stay in the denominator; they fall into "other"


def build(key, cfg, td, assignments, labels, umap):
    sub2type, type2lineage, _ = labels
    subtype = pd.read_csv(S.SUBTYPES, index_col=0)["cell_subtype"]
    celltype_of = lambda cell: sub2type.get(subtype.get(cell))
    group_of = (type2lineage.get if cfg["hierarchy"] == "lineage"
                else (lambda t: t))                 # cell_type: group is the type
    group_of_cell = lambda cell: group_of(celltype_of(cell))

    print(f"[{key}] scanning progenitor nodes + tracing {cfg['focus']} leaves ...",
          flush=True)
    fan = _trace(td, assignments, cfg, celltype_of, group_of)       # UMAP + subtrees
    e95 = [n for n in td.obst if n.startswith("E9.5")]              # composition:
    stypes = _source_types(td, e95, assignments, subtype, sub2type, cfg["exclude"])
    src_group = pd.Series(stypes).map(group_of)

    # highlighted source groups: explicit list, else the top N by node count.
    groups = cfg["groups"] or list(
        src_group.value_counts().head(S.N_FANIN_GROUPS).index)
    gcolors = S.group_colors(groups, cfg["palette"])

    # composition over the WHOLE fan-in, counting progenitor NODES: each highlighted
    # group keeps its colour, everything else folds into one grey "other" bar (sum 1).
    total = len(src_group)
    counts = src_group.value_counts()
    comp = pd.DataFrame({"src_group": groups})
    comp["count"] = comp["src_group"].map(counts).fillna(0).astype(int)
    comp["color"] = comp["src_group"].map(gcolors)
    other = int(total - comp["count"].sum())
    comp = pd.concat([comp, pd.DataFrame([{"src_group": "other", "count": other,
                     "color": S.GREY}])], ignore_index=True)
    comp["frac"] = comp["count"] / max(total, 1)
    comp.to_csv(os.path.join(S.NR, f"{key}_composition.csv"), index=False)

    # UMAP + subtrees are restricted to the highlighted source groups.
    fan = fan[fan["src_group"].isin(groups)]
    print(f"  {total:,} progenitor nodes ({other:,} other); {len(fan):,} leaf paths "
          f"from {len(groups)} kept groups", flush=True)
    fan = fan.merge(umap.rename(columns={"ux": "leaf_x", "uy": "leaf_y"}),
                    left_on="cell", right_index=True, how="inner")
    fan = fan.merge(umap.rename(columns={"ux": "anc_x", "uy": "anc_y"}),
                    left_on="anc_cell", right_index=True, how="inner")

    link = fan.sample(n=min(DISPLAY_LINES, len(fan)), random_state=int(S.SEED))
    link.to_pickle(os.path.join(S.NR, f"{key}_leaves.pkl"))

    # example subtrees: for each group, the largest subtree that group dominates
    # (>50% of leaves) with no flip-flopping root->leaf paths, so the focus leaf
    # sits among same-group cousins along lineages without flip-flops.
    lo, hi = S.SUBTREE_LEAF_RANGE
    examples = []
    for grp in groups:
        best = None
        for _, r in fan[fan["src_group"] == grp].iterrows():
            g = td.obst.get(r["tree"])
            if g is None or r["anc_node"] not in g:
                continue
            desc = nx.descendants(g, r["anc_node"]) | {r["anc_node"]}
            leaves = [n for n in desc if g.out_degree(n) == 0]
            if not (lo <= len(leaves) <= hi):
                continue
            fmatch = float(np.mean([group_of_cell(n) == grp for n in leaves]))
            if fmatch <= 0.5:
                continue
            amap = assignments.get(r["tree"], {})
            group_at = lambda n, a=amap, gg=g: group_of_cell(
                n if gg.out_degree(n) == 0 else a.get(n))
            if _oscillates(g, r["anc_node"], leaves, group_at):
                continue
            k = (len(leaves), fmatch)
            if best is None or k > (best[2], best[1]):
                best = (r, fmatch, len(leaves), g, desc)
        if best is None:
            print(f"  [{grp}] no clean majority-dominated subtree in range",
                  flush=True)
            continue
        r, fmatch, nleaf, g, desc = best
        sub = g.subgraph(desc)
        pos = _square_layout(sub, r["anc_node"])
        nodes = []
        for n in sub.nodes():
            is_leaf = g.out_degree(n) == 0
            cell = n if is_leaf else assignments.get(r["tree"], {}).get(n)
            grp_n = group_of_cell(cell) if cell and cell != "unknown" else None
            is_focus = is_leaf and celltype_of(n) == cfg["focus"]  # black, like UMAP
            nodes.append({
                "node": n, "x": pos[n][0], "y": pos[n][1], "is_leaf": is_leaf,
                "group": grp_n, "is_focus": is_focus,
                "color": S.INK if is_focus else gcolors.get(grp_n, S.GREY),
                "is_root": n == r["anc_node"]})
        examples.append({"src_group": grp, "root": r["anc_node"],
                         "tree": r["tree"], "n_leaves": nleaf,
                         "nodes": nodes, "edges": list(sub.edges())})
        print(f"  [{grp}] {r['tree']} leaves={nleaf} ({fmatch*100:.0f}% {grp})",
              flush=True)

    with open(os.path.join(S.NR, f"{key}_subtrees.pkl"), "wb") as f:
        pickle.dump(examples, f)
    print(f"  wrote {key}_leaves.pkl ({len(link):,}), {key}_composition.csv, "
          f"{key}_subtrees.pkl ({len(examples)})", flush=True)


def main(keys=None):
    import treedata
    labels = S.load_labels()
    umap = pd.read_csv(S.UMAP_CSV, index_col=0)
    umap.columns = ["ux", "uy"]
    with open(S.ASSIGN_PKL, "rb") as fh:
        assignments = pickle.load(fh)
    td = treedata.read_h5td(S.TREES)
    for key in (keys or list(S.FANIN)):
        build(key, S.FANIN[key], td, assignments, labels, umap)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
