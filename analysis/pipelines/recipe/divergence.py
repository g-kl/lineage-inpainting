"""loading + divergence-time helpers for the fate-specifying-TF recipe.

reads the control trees and the cell-type map; provides the per-leaf root->leaf
path and the snapped-name divergence time of a single knockout.
"""

import os
import glob
import pickle

import pandas as pd


def load_cell_type_map(data_dir, level="cell_type"):
    """snapped cell name -> group (level is a final_type_info.csv column, e.g.
    cell_type or lineage), via final_subtype.csv + final_type_info.csv."""
    sub = pd.read_csv(os.path.join(data_dir, "final_subtype.csv"), index_col=0)
    info = pd.read_csv(os.path.join(data_dir, "final_type_info.csv"), index_col=0)
    return {n: info.loc[st, level]
            for n, st in sub["cell_subtype"].items()
            if st in info.index and pd.notna(info.loc[st, level])}


def parent_map(tree):
    """node -> parent (None for roots)."""
    par = {}
    for n in tree.nodes():
        preds = list(tree.predecessors(n))
        par[n] = preds[0] if preds else None
    return par


def load_control(control_dir, name_to_ct):
    """per control tree: snapped names, node times, and for every leaf with a
    known cell type its (cell_type, root->leaf path)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(control_dir, "*.pkl"))):
        tn = os.path.splitext(os.path.basename(f))[0]
        tree = pickle.load(open(f, "rb"))["tree"]
        snap = {n: tree.nodes[n].get("snapped_name") for n in tree.nodes()}
        tnode = {n: tree.nodes[n].get("time") for n in tree.nodes()}
        par = parent_map(tree)
        leaves = {}
        for n in tree.nodes():
            if tree.out_degree(n) != 0:
                continue
            ct = name_to_ct.get(snap[n])
            if ct is None:
                continue
            path = [n]
            while par[path[-1]] is not None:
                path.append(par[path[-1]])
            path.reverse()                       # root -> leaf
            leaves[n] = (ct, path)
        out[tn] = {"snap": snap, "time": tnode, "leaves": leaves}
    return out


def first_divergence_time(path, ctrl_snap, ctrl_time, ptree):
    """control time of the first node on the root->leaf path that the knockout
    changed: pruned from the perturbed tree, or landed on a different snapped
    name.  None if the whole path is identical (leaf untouched)."""
    for n in path:
        if n not in ptree or ptree.nodes[n].get("snapped_name") != ctrl_snap[n]:
            return ctrl_time[n]
    return None
