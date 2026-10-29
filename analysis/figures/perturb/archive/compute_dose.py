"""row-2 panel-3 cache (no h5ad): Tbx5 dose-response cell-type abundance curves.

reads the per-dose lineage trees under analysis/results/dose_response/Tbx5_trees
(dose<NN>.pkl = Tbx5 held at NN% of its own expression).  for each dose we pool the
target-time on-manifold leaves over trees, map them to cell types, and take the
cell-type proportion.  normalised cell-type abundance = proportion / proportion at
the 100% (baseline) dose, so a depleted type drops below 1.  reads only the dose
pkls + the cell-type annotation CSVs (no h5ad).
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd

from analysis.pipelines.perturb.perturb_umap_lib import load_cell_categories, leaf_names
from analysis.figures.perturb import style as S


def dose_pct(path):
    """NN from dose<NN>.pkl (= % of own expression Tbx5 is held at)."""
    return int(os.path.splitext(os.path.basename(path))[0][len("dose"):])


def composition(names, name_to_ct, cell_types):
    counts = pd.Series(names).map(name_to_ct).value_counts()
    return np.array([counts.get(ct, 0) for ct in cell_types], dtype=float)


def main():
    name_to_ct, _, type_info = load_cell_categories(S.DATA_DIR)
    cell_types = sorted(type_info["cell_type"].dropna().unique())

    paths = sorted(glob.glob(os.path.join(S.DOSE_TREES, "dose*.pkl")), key=dose_pct)
    doses = np.array([dose_pct(p) for p in paths], dtype=float)
    print(f"{len(paths)} doses: {doses.tolist()}", flush=True)

    comps = []
    for p in paths:
        trees = pickle.load(open(p, "rb"))
        names = leaf_names(trees, S.TARGET_TIME, S.LEAF_TOL)
        comps.append(composition(names, name_to_ct, cell_types))
        print(f"  dose {dose_pct(p):3d}%: {len(names)} leaves", flush=True)
    comps = np.vstack(comps)                                   # (n_doses, n_types)

    props = comps / comps.sum(axis=1, keepdims=True)
    base = props[np.argmax(doses)]                             # 100% dose = baseline
    keep = base > 0
    cell_types = [ct for ct, k in zip(cell_types, keep) if k]
    abundance = (props[:, keep] / base[keep]).T                # (n_types, n_doses)

    # rank by depletion at the strongest dose (lowest % expression)
    strongest = int(np.argmin(doses))
    order = np.argsort(abundance[:, strongest])                # smallest abundance first
    top_idx = order[:4]
    print("top depleted:", [cell_types[i] for i in top_idx], flush=True)

    out = os.path.join(S.NUMERICAL, "dose.npz")
    np.savez(out, doses=doses, abundance=abundance, top_idx=top_idx,
             cell_types=np.array(cell_types, dtype=object))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
