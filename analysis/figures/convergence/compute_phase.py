#!/usr/bin/env python
"""panel e: both programs against time, per fate.

the endothelial panel keeps the two annotation groups. the two neuron panels are
binned by their trajectory's ~E8.5 positional score instead -- N_BINS equal-count
bins, edges taken over both neuron panels together so a bin means the same thing in
both, and so the hue is the same number panel d's colorbar carries.

    python analysis/figures/convergence/compute_phase.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import style as S


def kernel(t, Sc):
    """gaussian sufficient statistics on GRID for scores `Sc` of nodes at times `t`."""
    W = np.exp(-0.5 * ((S.GRID[:, None] - np.asarray(t)[None, :]) / S.BW) ** 2)
    return W.sum(1), W @ Sc


def score(Z, ax, sheet, panel):
    """x on the sheet's positional axis, y on the panel's own identity axis. the
    neuron panels share one positional axis so a bin, a hue and a positional score are
    all the same number across them; the fit population, and so the centring, is the
    same for every panel of a lineage."""
    mu = ax[f"{panel}_mu"]
    return np.stack([(Z - mu) @ ax[f"{sheet['axes']}_V"][:, 0],
                     (Z - mu) @ ax[f"{panel}_V"][:, 1]], 1)


def bin_edges(T):
    """equal-count bins of the positional score over the pooled neuron trajectories."""
    return np.quantile(T.x.values, np.linspace(0, 1, S.N_BINS + 1))


def groups(d, T, spec, edges):
    """per node, the group its trajectory belongs to. the annotation group comes from
    the cached trajectory table, the score bin from the ~E8.5 source."""
    if not spec["binned"]:
        return d.group.values
    t = T[T.panel == spec["panel"]]
    b = np.clip(np.digitize(t.x.values, edges[1:-1]), 0, S.N_BINS - 1)
    return d.traj.map(dict(zip(t.traj.values, [f"bin{i}" for i in b]))).fillna("").values


def curve(h, Sc):
    """gaussian kernel mean of both axes on the time grid, points with an effective
    sample size under MIN_ESS dropped."""
    w, s = kernel(h.time.values, Sc[h.index.values])
    M = s / np.maximum(w, 1e-12)[:, None]
    ok = w >= S.MIN_ESS
    return pd.DataFrame(dict(time=S.GRID[ok], ess=w[ok], x=M[ok, 0], y=M[ok, 1]))


def endpoint(h, Sc):
    """the mature state: E9.5 leaves averaged within tree, then over trees."""
    M = pd.DataFrame(Sc[h.index.values], index=h.tree.values).groupby(level=0).mean()
    return dict(time=S.T_LEAF, n=len(h), x=M[0].mean(), y=M[1].mean())


def halfway(v, t, end):
    """the time the curve is half of the way from where it starts to where the group's
    own E9.5 leaves are."""
    lvl = v[0] + 0.5 * (end - v[0])
    hit = np.flatnonzero(v >= lvl if end > v[0] else v <= lvl)
    if not len(hit):
        return np.nan
    i = hit[0]
    if i == 0:
        return t[0]
    sl = slice(i - 1, i + 1)
    o = 1 if end > v[0] else -1
    return np.interp(lvl, v[sl][::o], t[sl][::o])


def main():
    t0 = time.time()
    tic = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    tic("cached trajectories")
    d, Z = D.nodes()
    ax = D.axes()
    T = pd.read_csv(os.path.join(S.NR, "sources.csv"))
    edges = bin_edges(T[T.sheet == "neuro"])

    cur, end, meta = [], [], []
    for spec in S.PHASE:
        m = (d.panel == spec["panel"]).values
        h = d[m].reset_index(drop=True)
        Sc = score(Z[m], ax, S.SHEETS[spec["sheet"]], spec["panel"])
        g = groups(h, T, spec, edges)
        keys = ([f"bin{i}" for i in range(S.N_BINS)] if spec["binned"]
                else list(S.PROGRAMS[spec["panel"]]["x"]))
        for k in keys:
            sel = g == k
            c = curve(h[sel & (h.is_leaf == 0)], Sc)
            e = endpoint(h[sel & (h.is_leaf == 1)], Sc)
            cur.append(c.assign(panel=spec["panel"], group=k))
            end.append(dict(panel=spec["panel"], group=k, **e))
            meta.append(dict(
                panel=spec["panel"], group=k, n_traj=int(h[sel].traj.nunique()),
                n_node=int(sel.sum()), x_start=c.x.iloc[0], y_start=c.y.iloc[0],
                x_leaf=e["x"], y_leaf=e["y"],
                t_half_y=halfway(c.y.values, c.time.values, e["y"])))
        tic(f"    {spec['panel']}: {len(keys)} groups, "
            f"{int(pd.Series(g).isin(keys).sum()):,} of {len(h):,} nodes")

    for name, part in (("phase_curves", pd.concat(cur, ignore_index=True)),
                       ("phase_endpoints", pd.DataFrame(end)),
                       ("phase_groups", pd.DataFrame(meta))):
        part.to_csv(os.path.join(S.NR, f"{name}.csv"), index=False)
    B = pd.DataFrame(dict(group=[f"bin{i}" for i in range(S.N_BINS)],
                          lo=edges[:-1], hi=edges[1:]))
    n = T[T.sheet == "neuro"].copy()
    n["group"] = [f"bin{i}" for i in
                  np.clip(np.digitize(n.x.values, edges[1:-1]), 0, S.N_BINS - 1)]
    B = B.merge(n.groupby("group").agg(n=("traj", "size"), x_mean=("x", "mean"))
                .reset_index(), on="group", how="left")
    B.to_csv(os.path.join(S.NR, "phase_bins.csv"), index=False)

    for r in B.itertuples():
        print(f"      {r.group}  [{r.lo:+.2f}, {r.hi:+.2f}]  n {r.n:,}  "
              f"mean {r.x_mean:+.2f}", flush=True)
    for r in pd.DataFrame(meta).itertuples():
        print(f"      {r.panel:<4} {r.group:<14} {r.n_traj:>5,} traj  "
              f"x {r.x_start:+.2f} -> {r.x_leaf:+.2f}  "
              f"y {r.y_start:+.2f} -> {r.y_leaf:+.2f}  half at E{r.t_half_y:.2f}",
              flush=True)
    tic(f"wrote phase_curves/phase_endpoints/phase_groups/phase_bins.csv to {S.NR}")


if __name__ == "__main__":
    main()
