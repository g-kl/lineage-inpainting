"""bottom-row cache: per heart-screen gene, % depletion of each drawable heart
subtype (v2 annotation), plus a specificity ranking used to pick the most
subtype-specific knockouts.

depletion(gene, subtype) = 100 * (1 - perturbed/control on-manifold E9.0 readout
cells snapped to that subtype), pooled over trees.  specificity = normalised
Shannon entropy of the clipped per-subtype depletion (0 = a KO that depletes a
single subtype, 1 = a pan-heart depleter).  reads the heart-screen pkls + the v2
annotation csvs only (no h5ad).
"""

import os
import glob
import pickle
from collections import Counter

import numpy as np
import pandas as pd

from analysis.figures.perturb import style as S
from analysis.figures.perturb.compute_metrics import off_manifold_descendants
from utils import heart_diagram as HD


def drawable_subtypes():
    """v2 heart subtypes whose region carries a fillable path in the diagram."""
    return [st for st, r in S.HEART_SUBTYPE_TO_REGION.items()
            if any(rg == r and role == "fill" for rg, role in HD.REGION_OF.values())]


def subtype_counts(tree, name_to_st, keep):
    """Counter of on-manifold E9.0 readout cells snapped to each kept subtype."""
    bad = off_manifold_descendants(tree, S.OFF_RATIO)
    c = Counter()
    for n, d in tree.nodes(data=True):
        if n in bad:
            continue
        t = d.get("time")
        if t is None or abs(t - S.TARGET_TIME) >= S.LEAF_TOL:
            continue
        st = name_to_st.get(d.get("snapped_name"))
        if st in keep:
            c[st] += 1
    return c


def specificity(row):
    p = np.clip(row.values.astype(float), 0.0, None)
    tot = p.sum()
    if tot <= 0:
        return np.nan
    q = p[p > 0] / tot
    return float(-np.sum(q * np.log(q)) / np.log(len(p)))


def main():
    sub = pd.read_csv(S.SUBTYPE_V2, index_col=0)
    info = pd.read_csv(S.TYPE_INFO_V2, index_col=0)
    name_to_st = sub["cell_subtype"].to_dict()
    heart_ct = {l.lower() for l in S.heart_labels}
    keep = {s for s in info.index
            if str(info.loc[s, "cell_type"]).lower() in heart_ct}

    control = Counter()
    for f in sorted(glob.glob(os.path.join(S.HEART_SCREEN, "control", "*.pkl"))):
        control += subtype_counts(pickle.load(open(f, "rb"))["tree"], name_to_st, keep)

    pert = {}
    for f in sorted(glob.glob(os.path.join(S.HEART_SCREEN, "genes", "*.pkl"))):
        gene = os.path.splitext(os.path.basename(f))[0]
        c = Counter()
        for d in pickle.load(open(f, "rb")).values():
            c += subtype_counts(d["tree"], name_to_st, keep)
        pert[gene] = c
        print(f"  {gene}: {sum(c.values())} heart readout cells", flush=True)
    genes = sorted(pert)

    cols = [s for s in drawable_subtypes() if control.get(s, 0) >= S.HEART_MIN_CTRL_CELLS]
    dropped = [s for s in drawable_subtypes() if control.get(s, 0) < S.HEART_MIN_CTRL_CELLS]
    print(f"\ndrawable subtypes: {cols}")
    if dropped:
        print(f"dropped (< {S.HEART_MIN_CTRL_CELLS} control cells): {dropped}", flush=True)

    D = pd.DataFrame(
        [[100.0 * (1.0 - pert[g].get(s, 0) / control[s]) for s in cols] for g in genes],
        index=genes, columns=cols)
    D.index.name = "gene"

    rank = pd.DataFrame({
        "max_depl":     D.max(axis=1),
        "top_subtype":  D.idxmax(axis=1),
        "norm_entropy": D.apply(specificity, axis=1),
    }).sort_values("norm_entropy")
    rank.index.name = "gene"

    depl_out = os.path.join(S.NUMERICAL, "heart_subtype_depletion.csv")
    rank_out = os.path.join(S.NUMERICAL, "heart_specificity.csv")
    D.to_csv(depl_out)
    rank.to_csv(rank_out)
    print(f"wrote {depl_out}\nwrote {rank_out}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
