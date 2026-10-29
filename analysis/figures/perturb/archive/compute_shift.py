"""row-1 cache (no h5ad): grey backdrop + Nkx2-5 KO leaves split shifted / not.

every terminal leaf of the Nkx2-5 knockout trees (from the E9.5 HVG screen) carries a
snapped observed cell; the matching control tree (same topology, matching node names)
carries the cell that leaf landed on with no perturbation.  a target-time on-manifold
KO leaf is "shifted" when its snapped cell differs from control's at the same node.
on-manifold leaves are selected with S.OFF_RATIO (drops off-manifold subtrees).  we map
both groups to their reference UMAP coordinates (row 1 panel 2) and keep a global
subsample of all cells as the grey embryo backdrop shared by every UMAP panel.  reads
only the HVG screen pkls + final_umap.csv (no 55GB h5ad).
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd

from analysis.pipelines.perturb.perturb_result import off_manifold_descendants
from analysis.figures.perturb import style as S


def control_snapped(control_dir):
    """{tree_name: {node_name: snapped_name}} over all control trees."""
    out = {}
    for p in sorted(glob.glob(os.path.join(control_dir, "*.pkl"))):
        meta = pickle.load(open(p, "rb"))["meta"]
        tname = os.path.splitext(os.path.basename(p))[0]
        out[tname] = dict(zip(meta["node_name"], meta["snapped_name"]))
    return out


def shifted_split(gene_trees, ctrl_snap, umap):
    """UMAP coords of target-time on-manifold KO leaves, split shifted / not shifted."""
    shifted, unshifted = [], []
    for tname, d in gene_trees.items():
        meta = d["meta"]
        cs = ctrl_snap.get(tname, {})
        t = meta["time"].to_numpy(dtype=float)
        bad = off_manifold_descendants(d["tree"], S.OFF_RATIO)
        on = ~np.isin(meta["node_name"].to_numpy(), list(bad))
        keep = (np.abs(t - S.RING_TARGET_TIME) < S.LEAF_TOL) & on
        for node, snap in zip(meta["node_name"].to_numpy()[keep],
                              meta["snapped_name"].to_numpy()[keep]):
            if snap not in umap.index:
                continue
            xy = umap.loc[snap].to_numpy()
            (shifted if cs.get(node) != snap else unshifted).append(xy)
    to_arr = lambda v: np.vstack(v) if v else np.empty((0, 2))
    return to_arr(shifted), to_arr(unshifted)


def main():
    print("reading UMAP coordinates ...", flush=True)
    umap = pd.read_csv(os.path.join(S.DATA_DIR, "final_umap.csv"), index_col=0)

    rng = np.random.default_rng(S.seed)
    backdrop = umap.iloc[rng.choice(len(umap), min(S.N_BACKDROP, len(umap)), replace=False)]
    backdrop = backdrop[["UMAP1", "UMAP2"]].to_numpy(float)

    ctrl_snap = control_snapped(os.path.join(S.RING_SCREEN, "control"))
    with open(os.path.join(S.RING_SCREEN, "genes", f"{S.HERO_GENE}.pkl"), "rb") as fh:
        gene_trees = pickle.load(fh)
    print(f"  {len(ctrl_snap)} control trees, {len(gene_trees)} {S.HERO_GENE} KO trees",
          flush=True)

    shifted, unshifted = shifted_split(gene_trees, ctrl_snap, umap)
    print(f"  {S.HERO_GENE} KO leaves: {len(shifted)} shifted, {len(unshifted)} unshifted",
          flush=True)

    out = os.path.join(S.NUMERICAL, "shift.npz")
    np.savez(out, backdrop=backdrop, shifted=shifted, unshifted=unshifted)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
