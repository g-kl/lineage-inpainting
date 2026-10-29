"""every (annotated gene, cell type) pair scored by model depletion and by E9.0 mean
expression -> numerical_results/pairs.csv, pairs.md.

the universe is the MGI-annotated genes x the cell types an association can point at (the
phenotype map's, with more than MIN_CTRL control cells). the MGI pairs are the ones an
association links. the association set comes from the perturb figure's compute_mgi_box.py.
depletion is 1 - perturbed / control readout leaves per cell type at E9.0, both read under
the Tbx5 figure's protocol, freeze at 1.1 (readout.py).

    <rapids_singlecell python> -m analysis.figures.supp_mgi.compute_pairs   # cuml knn
"""
import os
import pickle

import pandas as pd

from analysis.figures.perturb.compute_mgi_box import (
    CONCEPT_TO_COARSE, MIN_CTRL, PHENO_MAP, load_associations)
from analysis.figures.perturb.readout import name_to_celltype
from analysis.figures.supp_mgi import style as S
from analysis.figures.supp_mgi.readout import cell_types, snap_forward, tally


def main():
    name2ct = name_to_celltype()
    genes_dir = os.path.join(S.RING_SCREEN, "genes")
    gene_by_upper = {os.path.splitext(e.name)[0].upper(): os.path.splitext(e.name)[0]
                     for e in os.scandir(genes_dir) if e.name.endswith(".pkl")}

    ctrl = tally([pickle.load(open(e.path, "rb")) for e in
                  sorted(os.scandir(os.path.join(S.RING_SCREEN, "control")),
                         key=lambda e: e.name)], [S.OFF_RATIO])[S.OFF_RATIO]

    pm = pd.read_csv(PHENO_MAP).dropna(subset=["cell_type"])
    reachable = set(pm.loc[pm["concept_key"].isin(CONCEPT_TO_COARSE), "cell_type"])
    assoc = load_associations(set(gene_by_upper))
    per = {}
    for gu in sorted(assoc.loc[assoc["Gene"].isin(gene_by_upper), "Gene"].unique()):
        g = gene_by_upper[gu]
        trees = pickle.load(open(os.path.join(genes_dir, f"{g}.pkl"), "rb"))
        per[g] = tally(list(trees.values()), [S.OFF_RATIO])[S.OFF_RATIO]
    fwd = (snap_forward(set(ctrl[1]).union(*(t[1] for t in per.values())))
           if S.MODE == "freeze" else {})
    control = cell_types(ctrl, S.MODE, fwd, name2ct)
    pert = {g: cell_types(t, S.MODE, fwd, name2ct) for g, t in per.items()}

    cts = sorted(c for c in reachable if control.get(c, 0) > MIN_CTRL)
    expr = pd.read_parquet(os.path.join(S.NR, "expression.parquet"))
    no_expr = [c for c in cts if c not in expr.index]
    if no_expr:
        print(f"no E{S.RING_TARGET_TIME:g} reference cells for {no_expr}, dropped", flush=True)
        cts = [c for c in cts if c in expr.index]
    assoc = assoc[assoc["Gene"].isin(gene_by_upper) & assoc["cell_type"].isin(cts)]
    coarse = assoc.groupby(["Gene", "cell_type"])["coarse"].agg(
        lambda s: "; ".join(sorted(s))).to_dict()

    rows = []
    for gu in sorted(assoc["Gene"].unique()):
        g = gene_by_upper[gu]
        for ct in cts:
            rows.append({"gene": g, "cell_type": ct, "is_mgi": (gu, ct) in coarse,
                         "coarse": coarse.get((gu, ct), ""),
                         "depletion": 1.0 - pert[g].get(ct, 0) / control[ct],
                         "expression": expr.at[ct, g]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(S.NR, "pairs.csv"), index=False)

    mgi = df["is_mgi"]
    lines = [
        "# MGI pairs: the universe", "",
        f"{len(df):,} pairs = {df['gene'].nunique()} annotated genes x {len(cts)} cell types, "
        f"{mgi.sum():,} MGI pairs. depletion at off-manifold ratio {S.OFF_RATIO:.1f} under "
        f"{S.MODE}, expression the E{S.RING_TARGET_TIME:.1f} mean log1p CP10K.", "",
        f"of the MGI pairs, {(df.loc[mgi, 'expression'] == 0).mean():.3f} have zero "
        f"expression, {(df.loc[mgi, 'depletion'] >= 1).mean():.3f} full depletion and "
        f"{(df.loc[mgi, 'depletion'] < -1).mean():.3f} depletion below -1.", "",
        "| cell type | MGI pairs |", "| --- | --- |",
        *(f"| {ct} | {n} |" for ct, n in
          df[mgi].groupby("cell_type").size().sort_values(ascending=False).items()),
        "",
    ]
    out = os.path.join(S.NR, "pairs.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines[:5]), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
