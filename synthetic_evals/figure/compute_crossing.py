"""crossing high-dim eval pkls -> crossing_recon.pkl + crossing_fate.csv.

reads high_dim_crossing/results_<exp>_d*.pkl. fate accuracy is per parent, every
transition x dim (the main page plots the second split, the supplement all of them);
the recon states are projected to 1-d.

    python -m synthetic_evals.figure.compute_crossing
"""
import os
import pickle

import numpy as np
import pandas as pd

from synthetic_evals.figure import style as S

VARIANTS = ("noncrossing", "crossing")   # coherent, then crossing

# the attractors on the latent axis at each target time: the same set in both variants,
# only the parent -> child wiring differs (config/dataset/high_dim_crossing/).
FATES = {6.0: (-1.0, 1.0), 7.0: (-1.0, 1.0),
         8.0: (-1.5, -0.5, 0.5, 1.5), 9.0: (-1.5, -0.5, 0.5, 1.5)}


def _pkl(dim):
    return os.path.join(S.HD_DIR, f"results_{S.CROSS_EXP}_d{dim}.pkl")


def _proj(A, w):
    """project a state array (..., d) onto the recorded latent direction -> (..., 1)."""
    return (np.asarray(A, np.float32) @ w)[..., None]


def _fate_acc(ev, cache_key, pt):
    """fate accuracy at one transition: every true and predicted child is called to its
    nearest attractor on the latent axis, and each parent scores 1 - TV between its
    predicted and true sibling fate histograms (for a one-fate parent, the fraction of
    predictions on it). mean over parents."""
    w = np.asarray(ev["proj"]["w"], np.float64)
    mu = np.asarray(FATES[pt[1]])

    def call(A):
        return np.abs((np.asarray(A, np.float64) @ w)[..., None] - mu).argmin(-1)

    true, pred = call(ev["gt_pairs"][pt]["x_tgt"]), call(ev[cache_key][pt])   # (M,), (S, M)
    c2p = ev["pool_pairs"][pt]["child_to_parent"]
    accs = []
    for p in np.unique(c2p):
        ht = np.bincount(true[c2p == p], minlength=len(mu)) / (c2p == p).sum()
        hp = np.bincount(pred[:, c2p == p].ravel(), minlength=len(mu)) / pred[:, c2p == p].size
        accs.append(1.0 - 0.5 * np.abs(ht - hp).sum())
    return float(np.mean(accs))


def _trim(ev, methods):
    """the minimal ev subset the forward recon drawers read, states projected to 1-d."""
    w = np.asarray(ev["proj"]["w"], np.float32)
    keys = [S.RECON_CACHE_KEY[m] for m in methods]
    out = {
        "obs_times": list(ev["obs_times"]),
        "pair_times": list(ev["pair_times"]),
        "trajectories": [(np.asarray(t, np.float32), float(mu)) for t, mu in ev["trajectories"]],
        "gt_pairs": {}, "pool_pairs": {},
        **{ck: {} for ck in keys},
    }
    for pt in ev["pair_times"]:
        out["gt_pairs"][pt] = {"x_src": _proj(ev["gt_pairs"][pt]["x_src"], w),
                               "x_tgt": _proj(ev["gt_pairs"][pt]["x_tgt"], w)}
        out["pool_pairs"][pt] = {
            "child_to_parent": np.asarray(ev["pool_pairs"][pt]["child_to_parent"])}
        for ck in keys:
            out[ck][pt] = _proj(ev[ck][pt], w)
    return out


def main():
    with open(_pkl(S.RECON_DIM), "rb") as f:
        one = pickle.load(f)

    # only the methods actually present in the pkl (baselines can be turned off upstream)
    methods = [m for m in S.METHODS
               if S.RECON_CACHE_KEY[m] in one[next(iter(one))]]
    recon = {v: _trim(one[v], methods) for v in VARIANTS if v in one}
    with open(os.path.join(S.NR, "crossing_recon.pkl"), "wb") as f:
        pickle.dump(recon, f)

    fate_rows = []
    for d in S.DIMS:
        if not os.path.exists(_pkl(d)):
            continue
        with open(_pkl(d), "rb") as f:
            res = pickle.load(f)
        for v in VARIANTS:
            if v not in res:
                continue
            for m in methods:
                for pt in res[v]["pair_times"]:
                    fate_rows.append({"variant": v, "method": m, "dim": d,
                                      "t_src": pt[0], "t_tgt": pt[1],
                                      "acc": _fate_acc(res[v], S.RECON_CACHE_KEY[m], pt)})
    pd.DataFrame(fate_rows).to_csv(os.path.join(S.NR, "crossing_fate.csv"), index=False)

    print(f"wrote crossing caches -> {S.NR}  ({len(recon)} variants, "
          f"{len(fate_rows)} fate rows, methods={methods})")


if __name__ == "__main__":
    main()
