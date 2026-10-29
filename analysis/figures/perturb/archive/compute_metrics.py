"""row-2 cache: per heart-screen gene, heart/total depletion (100% & 50%) + shift time.

depletion  = 1 - perturbed/control on-manifold leaves at the readout (heart cells
             for %heart, all cells for %total), pooled over trees.  computed once
             for the 100%-knockdown (full knockout) screen and once for its
             50%-knockdown counterpart (the row-2 100%-vs-50% scatter).
shift time = mean over PERTURBED readout leaves of the divergence onset (100% screen
             only).  control and knockout share the tree topology (node names match),
             so a leaf is "perturbed" when its snapped cell differs from control's at
             that node; walking that leaf up to the root, the shift time is the time of
             the most ancestral node still differing from control (the first divergence
             before the two trajectories reconverge).
reads only the heart-screen pkls + the cell-type annotation CSVs (no h5ad).
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd
import networkx as nx

from analysis.figures.perturb import style as S


def name_to_subtype_map():
    sub = pd.read_csv(os.path.join(S.DATA_DIR, "final_subtype.csv"), index_col=0)
    info = pd.read_csv(os.path.join(S.DATA_DIR, "final_type_info.csv"), index_col=0)
    heart = {s for s in info.index if info.loc[s, "cell_type"] in S.heart_labels}
    return sub["cell_subtype"].to_dict(), heart


def off_manifold_descendants(tree, ratio):
    bad = set()
    for n, d in tree.nodes(data=True):
        ld = d.get("local_dist")
        if d.get("changed") and ld is not None and d.get("proj_dist", 0.0) > ratio * ld:
            bad.add(n)
            bad.update(nx.descendants(tree, n))
    return bad


def counts_at_target(tree, name_to_st, heart_set):
    """(n_total, n_heart) on-manifold readout cells, off-manifold subtrees dropped."""
    bad = off_manifold_descendants(tree, S.OFF_RATIO)
    n_total = n_heart = 0
    for n, d in tree.nodes(data=True):
        if n in bad:
            continue
        t = d.get("time")
        if t is None or abs(t - S.TARGET_TIME) >= 0.25:
            continue
        n_total += 1
        if name_to_st.get(d.get("snapped_name")) in heart_set:
            n_heart += 1
    return n_total, n_heart


def divergence_shifts(ctrl_tree, pert_tree):
    """divergence-onset time for each perturbed control readout leaf.

    control and knockout share the topology (matching node names).  iterating over
    the control readout leaves, a leaf is perturbed when, in the knockout, it went
    off-manifold (its subtree was pruned) or its snapped cell differs from control.
    walking such a leaf up to the root, the shift time is the time of the most
    ancestral node still diverged (stop at the first node that reconverges to the
    control state).  nan only if no control leaf is perturbed.
    """
    cs = {n: ctrl_tree.nodes[n].get("snapped_name") for n in ctrl_tree.nodes()}
    ps = {n: pert_tree.nodes[n].get("snapped_name") for n in pert_tree.nodes()}
    bad = off_manifold_descendants(pert_tree, S.OFF_RATIO)
    diverged = lambda n: n in bad or (cs.get(n) is not None and ps.get(n) != cs.get(n))
    out = []
    for n in ctrl_tree.nodes():
        if ctrl_tree.out_degree(n) != 0:                    # leaves only
            continue
        t = ctrl_tree.nodes[n].get("time")
        if t is None or abs(t - S.TARGET_TIME) >= 0.25:
            continue
        if not diverged(n):
            continue                                        # unperturbed leaf
        onset, m = t, n
        while m is not None and diverged(m):
            onset = ctrl_tree.nodes[m].get("time", onset)   # still diverged here
            preds = list(ctrl_tree.predecessors(m))
            m = preds[0] if preds else None
        out.append(onset)
    return out


def screen_metrics(screen_dir, name_to_st, heart_set, with_shift):
    """{gene: {heart_depl, total_depl, [shift_time]}} for one heart screen dir."""
    ctrl_trees = {}
    ctrl_total = ctrl_heart = 0
    for p in sorted(glob.glob(os.path.join(screen_dir, "control", "*.pkl"))):
        tree = pickle.load(open(p, "rb"))["tree"]
        ctrl_trees[os.path.splitext(os.path.basename(p))[0]] = tree
        nt, nh = counts_at_target(tree, name_to_st, heart_set)
        ctrl_total += nt
        ctrl_heart += nh
    print(f"  control: total={ctrl_total} heart={ctrl_heart}", flush=True)

    out = {}
    for f in sorted(glob.glob(os.path.join(screen_dir, "genes", "*.pkl"))):
        gene = os.path.splitext(os.path.basename(f))[0]
        pert_total = pert_heart = 0
        shifts = []
        for tname, d in pickle.load(open(f, "rb")).items():
            nt, nh = counts_at_target(d["tree"], name_to_st, heart_set)
            pert_total += nt
            pert_heart += nh
            if with_shift and tname in ctrl_trees:
                shifts.extend(divergence_shifts(ctrl_trees[tname], d["tree"]))
        rec = {"heart_depl": 100 * (1 - pert_heart / ctrl_heart),
               "total_depl": 100 * (1 - pert_total / ctrl_total)}
        if with_shift:
            rec["shift_time"] = float(np.mean(shifts)) if shifts else np.nan
        out[gene] = rec
        msg = f"    {gene}: heart={rec['heart_depl']:.0f}% total={rec['total_depl']:.0f}%"
        if with_shift:
            msg += f" shift={rec['shift_time']:.2f}"
        print(msg, flush=True)
    return out


def main():
    name_to_st, heart_set = name_to_subtype_map()

    print("100% knockdown screen:", flush=True)
    full = screen_metrics(S.HEART_SCREEN, name_to_st, heart_set, with_shift=True)
    print("50% knockdown screen:", flush=True)
    half = screen_metrics(S.HEART_SCREEN_50, name_to_st, heart_set, with_shift=False)

    rows = []
    for gene, rec in full.items():
        h = half.get(gene, {})
        rows.append({"gene": gene,
                     "heart_depl": rec["heart_depl"],
                     "total_depl": rec["total_depl"],
                     "shift_time": rec["shift_time"],
                     "heart_depl_50": h.get("heart_depl", np.nan),
                     "total_depl_50": h.get("total_depl", np.nan)})

    out = os.path.join(S.NUMERICAL, "metrics.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
