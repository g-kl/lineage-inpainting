"""the screen's E9.0 readout under an off-manifold protocol. the Tbx5 figure's recipe
(analysis/pipelines/tbx5/support_comparison/predicted.py), read at E9.0 rather than E9.5.

a readout leaf is off the manifold once a changed node on its root path projected farther
than the ratio x its local spacing (perturb_result.off_manifold_descendants). an on-manifold
leaf contributes the cell it snapped to. an off one is dropped under `prune`, and under
`freeze` contributes the snapped cell of its last on-manifold ancestor, carried forward to
its nearest E9.0 devmap cell in the devmap fit's own scVI latent (cuml brute knn).
"""
from collections import Counter

import numpy as np
import pandas as pd

from analysis.figures.supp_mgi import style as S
from analysis.pipelines.perturb.perturb_result import off_manifold_descendants
from analysis.pipelines.tbx5.support_comparison import common as C
from analysis.pipelines.tbx5.support_comparison.predicted import last_on_manifold

STAGE = f"E{S.RING_TARGET_TIME:.1f}"   # the devmap cells a frozen state is carried onto


def tally(trees, ratios):
    """{ratio: (on, frozen)} over the readout leaves of a list of {"meta", "tree"}: on
    counts the on-manifold leaves' snapped cells, frozen the off-manifold leaves' last
    on-manifold ancestors' snapped cells."""
    res = {r: (Counter(), Counter()) for r in ratios}
    for d in trees:
        meta = d["meta"]
        tw = np.abs(meta["time"].to_numpy(float) - S.RING_TARGET_TIME) < S.RING_TOL
        node, snap = meta["node_name"].to_numpy(), meta["snapped_name"].to_numpy()
        for r in ratios:
            bad = off_manifold_descendants(d["tree"], r)
            on = np.fromiter((n not in bad for n in node), bool, len(node))
            res[r][0].update(snap[tw & on])
            if (tw & ~on).any():
                rep = last_on_manifold(d["tree"], bad)
                res[r][1].update(rep[n] for n in node[tw & ~on] if rep.get(n) is not None)
    return res


def snap_forward(cells):
    """pre-E9.0 cells -> their nearest E9.0 devmap cell in the devmap latent
    (predicted.to_e95 at E9.0)."""
    early = pd.Index(sorted({c for c in cells if not c.startswith(STAGE)}))
    if not len(early):
        return {}
    idx = C.devmap_index()
    missing = early.difference(idx)
    assert not len(missing), f"{len(missing)} frozen states are not devmap cells"
    late = idx[idx.str.startswith(STAGE)]
    j = C.knn(C.devmap_latent(late), C.devmap_latent(early), 1).ravel()
    print(f"snapped {len(early):,} frozen states onto {STAGE} cells", flush=True)
    return dict(zip(early, late[j]))


def cell_types(tal, mode, fwd, name2ct):
    """cell type -> readout leaves under one protocol, from one tally entry."""
    on, frozen = tal
    c = Counter()
    for cell, k in on.items():
        c[name2ct.get(cell)] += k
    if mode == "freeze":
        for cell, k in frozen.items():
            c[name2ct.get(fwd.get(cell, cell))] += k
    return c
