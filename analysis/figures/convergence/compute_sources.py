#!/usr/bin/env python
"""every trajectory's ~E8.5 source and its E9.5 leaf: umap coordinates for both, and
both scored on the sheet's two program axes. also the subsample of links panel d
draws and the colour limit panels d and e share.

    python analysis/figures/convergence/compute_sources.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import style as S

LINK_COLS = ("sheet", "panel", "x", "src_u1", "src_u2", "leaf_u1", "leaf_u2")


def main():
    t0 = time.time()
    tic = lambda m: print(f"[{time.time() - t0:6.1f}s] {m}", flush=True)

    tic("cached trajectories and axes")
    d, Z = D.nodes()
    ax = D.axes()
    name, ctype = D.obs()
    u = D.umap()
    rng = np.random.default_rng(S.SEED)

    out, links, norm, funnel = [], [], [], []
    for key, sheet in S.SHEETS.items():
        V, mu = ax[f"{sheet['axes']}_V"], ax[f"{sheet['axes']}_mu"]
        T, fn = D.sheet_table(d, Z, sheet, name, ctype, V, mu)
        T = T.assign(sheet=key)
        for pre, col in (("src", "src_cell"), ("leaf", "leaf_cell")):
            T[[f"{pre}_u1", f"{pre}_u2"]] = u.reindex(T[col].values).values
        L = T.iloc[rng.choice(len(T), min(S.N_LINK, len(T)), replace=False)]
        links.append(L[list(LINK_COLS)])
        norm.append(dict(sheet=key, vmax=np.abs(np.quantile(L.x, [0.02, 0.98])).max()))
        out.append(T)
        funnel.append(dict(sheet=key, **fn, classifiable=(T["bin"] != "").mean()))
        tic(f"    {key}: {fn['traj']:,} trajectories, {fn['anchored']:,} with an "
            f"E{S.T_SRC} anchor, minus {fn['wrong_germ']:,} not {sheet['germ']} and "
            f"{fn['at_dest']:,} already at the destination -> {fn['kept']:,} "
            f"({fn['kept'] / fn['traj']:.1%}), "
            f"{(T['bin'] != '').mean():.1%} classifiable, "
            f"umap coords {T.src_u1.notna().mean():.1%}")
    T = pd.concat(out, ignore_index=True)
    T.to_csv(os.path.join(S.NR, "sources.csv"), index=False)
    pd.concat(links, ignore_index=True).to_csv(
        os.path.join(S.NR, "umap_links.csv"), index=False)
    pd.DataFrame(norm).to_csv(os.path.join(S.NR, "score_norm.csv"), index=False)
    pd.DataFrame(funnel).to_csv(os.path.join(S.NR, "funnel.csv"), index=False)

    top = (T.groupby(["sheet", "src_type"]).agg(n=("traj", "size"), x=("x", "mean"))
           .reset_index().sort_values(["sheet", "n"], ascending=[True, False]))
    top.to_csv(os.path.join(S.NR, "src_types.csv"), index=False)
    for key, h in top.groupby("sheet"):
        print(f"      {key} source types ({len(h)} in all):", flush=True)
        for r in h.head(S.N_TYPES + 2).itertuples():
            print(f"        {r.src_type:<30} {r.n:>6,}  mean x {r.x:+.2f}", flush=True)
    tic("wrote sources.csv, umap_links.csv, score_norm.csv, funnel.csv, "
        f"src_types.csv to {S.NR}")


if __name__ == "__main__":
    main()
