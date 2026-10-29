#!/usr/bin/env python
"""row 1: per CHD panel gene, how much of the E9.0 readout the knockout shifted, and
when it did it.

shifted     100 * (1 - perturbed/control on-manifold readout cells), pooled over trees,
            over the heart cells and over all of them. computed for the full knockout
            and for its 50% knockdown counterpart.
shift time  mean over the perturbed readout leaves of the divergence onset. control and
            knockout share a topology, so a leaf is perturbed when the knockout went
            off the manifold there or snapped it somewhere else; walking that leaf up,
            the onset is the time of the most ancestral node still diverged.

reads the two screens' pkls and the annotation only.

    python compute_metrics.py
"""
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as S
import data
from analysis.pipelines.perturb.perturb_result import off_manifold_descendants


def heart_cells():
    """cell name -> subtype, and the subtypes in the heart."""
    sub2type, _ = data.type_map()
    heart = {s for s, t in sub2type.items() if t in data.HEART_TYPES}
    return data.devmap_subtypes().to_dict(), heart


def counts(tree, name_to_st, heart):
    """(all, heart) on-manifold readout cells, off-manifold subtrees dropped."""
    bad = off_manifold_descendants(tree, S.CHD_OFF_RATIO)
    n = h = 0
    for x, d in tree.nodes(data=True):
        t = d.get("time")
        if x in bad or t is None or abs(t - S.CHD_TIME) >= S.CHD_TOL:
            continue
        n += 1
        h += name_to_st.get(d.get("snapped_name")) in heart
    return n, h


def onsets(ct, pt):
    """divergence onset per perturbed control readout leaf."""
    bad = off_manifold_descendants(pt, S.CHD_OFF_RATIO)
    cs = {n: ct.nodes[n].get("snapped_name") for n in ct}
    div = lambda n: n in bad or (cs.get(n) is not None
                                and pt.nodes[n].get("snapped_name") != cs.get(n))
    out = []
    for n in ct:
        t = ct.nodes[n].get("time")
        if ct.out_degree(n) or t is None or abs(t - S.CHD_TIME) >= S.CHD_TOL \
                or not div(n):
            continue
        onset, m = t, n
        while m is not None and div(m):
            onset = ct.nodes[m].get("time", onset)
            m = next(iter(ct.predecessors(m)), None)
        out.append(onset)
    return out


def one_screen(path, name_to_st, heart, with_onset):
    """{gene: {heart, total, [shift_time]}} for one screen dir."""
    ctrl, c_all, c_heart = {}, 0, 0
    for p in sorted(glob.glob(os.path.join(path, "control", "*.pkl"))):
        with open(p, "rb") as fh:
            tree = pickle.load(fh)["tree"]
        ctrl[os.path.splitext(os.path.basename(p))[0]] = tree
        n, h = counts(tree, name_to_st, heart)
        c_all, c_heart = c_all + n, c_heart + h
    print(f"  control: {c_all:,} readout cells, {c_heart:,} in the heart", flush=True)

    out = {}
    for f in sorted(glob.glob(os.path.join(path, "genes", "*.pkl"))):
        with open(f, "rb") as fh:
            trees = pickle.load(fh)
        p_all, p_heart, on = 0, 0, []
        for name, d in trees.items():
            n, h = counts(d["tree"], name_to_st, heart)
            p_all, p_heart = p_all + n, p_heart + h
            if with_onset and name in ctrl:
                on.extend(onsets(ctrl[name], d["tree"]))
        rec = {"heart_shifted": 100 * (1 - p_heart / c_heart),
               "total_shifted": 100 * (1 - p_all / c_all)}
        if with_onset:
            rec["shift_time"] = float(np.mean(on)) if on else np.nan
        out[os.path.splitext(os.path.basename(f))[0]] = rec
    return out


def main():
    name_to_st, heart = heart_cells()
    print("full knockout:", flush=True)
    full = one_screen(S.CHD_SCREEN, name_to_st, heart, True)
    print("50% knockdown:", flush=True)
    half = one_screen(S.CHD_SCREEN_50, name_to_st, heart, False)

    d = pd.DataFrame(full).T
    d["heart_shifted_50"] = pd.DataFrame(half).T.heart_shifted
    d["specificity"] = np.log2(d.heart_shifted.clip(lower=1)
                               / d.total_shifted.clip(lower=1))
    d.index.name = "gene"
    d.to_csv(f"{S.NR}/metrics.csv")
    print(f"\n=== {len(d)} genes, % of the readout shifted", flush=True)
    print(d.sort_values("heart_shifted", ascending=False).round(2).to_string(),
          flush=True)


if __name__ == "__main__":
    main()
