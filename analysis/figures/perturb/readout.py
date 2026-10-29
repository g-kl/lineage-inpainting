"""shared readout helpers for the ring-screen validation panels (b MGI, c gnomad).

both panels score a knockout by pooling its perturbed trees' on-manifold readout
leaves near the target time and comparing the per-cell-type counts to control.  the
tree walk + off-manifold pruning + cell-type map live here so compute_mgi_box.py and
compute_gnomad_box.py share one copy.
dataset-free: reads only the screen pkls + the annotation csvs.
"""

import os

import numpy as np
import pandas as pd
import networkx as nx

from analysis.figures.perturb import style as S


def name_to_celltype():
    """cell subtype -> coarse cell type, from the final annotation csvs."""
    sub = pd.read_csv(os.path.join(S.DATA_DIR, "final_subtype.csv"), index_col=0)
    info = pd.read_csv(os.path.join(S.DATA_DIR, "final_type_info.csv"), index_col=0)
    return sub["cell_subtype"].map(info["cell_type"]).to_dict()


def readout_scores(tree, name2ct, target_time, tol):
    """(off-manifold score, cell type) for each readout node near target_time.

    the score is the max proj/local-dist ratio along the root->node path (only
    changed nodes count), so thresholding it prunes a node and all its descendants
    once the subtree leaves the manifold.
    """
    score, out = {}, []
    for n in nx.topological_sort(tree):
        d = tree.nodes[n]
        ld = d.get("local_dist")
        s = d.get("proj_dist", 0.0) / ld if (d.get("changed") and ld) else float("-inf")
        preds = list(tree.predecessors(n))
        score[n] = max(s, score[preds[0]]) if preds else s
        t = d.get("time")
        if t is not None and abs(t - target_time) < tol:
            out.append((score[n], name2ct.get(d.get("snapped_name"))))
    return out


def counts_at_ratio(readouts, ratio):
    """on-manifold per-cell-type counts (nodes with score <= ratio)."""
    counts = {}
    for s, ct in readouts:
        if s <= ratio:
            counts[ct] = counts.get(ct, 0) + 1
    return counts


def effect_norm(counts, ctrl):
    """l2 norm of the per-cell-type abundance log2fc vector vs control."""
    keys = set(counts) | set(ctrl)
    return float(np.linalg.norm([np.log2((counts.get(k, 0) + 1) / (ctrl.get(k, 0) + 1))
                                 for k in keys]))
