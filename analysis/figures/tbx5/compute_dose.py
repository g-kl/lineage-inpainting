#!/usr/bin/env python
"""row 1: the Tbx5 dose response.

one tree set per dose (dose<NN>.pkl holds Tbx5 at NN% of its own expression), every
dose grown over the same trees, so a node is the same trajectory at every dose. per
dose and per cell subtype, the share of that subtype's E9.0 readout trajectories the
knockdown has shifted - snapped somewhere else than it was at the 100% dose, or taken
off the manifold - which is what row 2's curves count. a
trajectory is scored under the subtype its own 100% dose cell carries, so the
denominator is the same at every dose and nothing has to stand in for a leaf that
never formed.

    python compute_dose.py
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


def dose_pct(path):
    return int(os.path.splitext(os.path.basename(path))[0][len("dose"):])


def readout(trees):
    """(snapped cell, off-manifold) over the readout leaves of one tree set, keyed by
    tree and node so the doses line up trajectory by trajectory."""
    out = []
    for name, d in trees.items():
        meta, tree = d["meta"], d["tree"]
        bad = off_manifold_descendants(tree, S.CHD_OFF_RATIO)
        m = meta.loc[np.abs(meta.time.to_numpy(float) - S.CHD_TIME) < S.CHD_TOL,
                     ["node_name", "snapped_name"]].copy()
        m["off"] = np.fromiter((n in bad for n in m.node_name), bool, len(m))
        m["tree"] = name
        out.append(m)
    r = pd.concat(out, ignore_index=True).set_index(["tree", "node_name"])
    assert r.index.is_unique, "a node is named twice in its own tree"
    return r


def main():
    paths = sorted(glob.glob(os.path.join(S.DOSE_TREES, "dose*.pkl")), key=dose_pct)
    per = {}
    for p in paths:
        with open(p, "rb") as fh:
            per[dose_pct(p)] = readout(pickle.load(fh))
        r = per[dose_pct(p)]
        print(f"  dose {dose_pct(p):3d}%: {len(r):,} readout leaves, "
              f"{int(r.off.sum()):,} off-manifold", flush=True)

    full = per[max(per)]                   # the 100% dose is the unperturbed side
    assert not full.off.any(), "the 100% dose leaves the manifold somewhere"
    who = full.snapped_name.map(data.devmap_subtypes()).to_numpy()

    rows = {}
    for k, r in sorted(per.items()):
        r = r.reindex(full.index)
        assert r.snapped_name.notna().all(), "the doses do not share their trajectories"
        moved = (r.off.to_numpy(bool)
                 | (r.snapped_name.to_numpy() != full.snapped_name.to_numpy()))
        rows[k] = pd.Series(moved).groupby(who).mean()

    d = pd.DataFrame(rows).T.sort_index()  # dose x subtype
    d.index.name = "dose"
    n = pd.Series(who).value_counts()
    a = d.loc[:, n.reindex(d.columns).to_numpy() >= S.MIN_DOSE_LEAVES]
    a.to_csv(f"{S.NR}/dose.csv")

    high = a.loc[a.index.min()].sort_values(ascending=False)
    print(f"\n=== {len(a.columns)} of {d.shape[1]} subtypes carry the "
          f"{S.MIN_DOSE_LEAVES} readout trajectories it takes to be drawn, over "
          f"{len(a)} doses", flush=True)
    print(pd.DataFrame({"leaves": n.reindex(high.index[:S.N_DOSE]),
                        f"shifted at {a.index.min()}%": high.head(S.N_DOSE).round(3)})
          .to_string(), flush=True)
    print(a[high.index[:S.N_DOSE]].round(2).to_string(), flush=True)


if __name__ == "__main__":
    main()
