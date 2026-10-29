"""compute the MGI perturbed-fraction box panel -> numerical_results/mgi_box.{csv,npz}.

each MGI association is a screened gene whose curated phenotype maps to a cell type;
it is scored by the fraction of that cell type's E9.0 control readout leaves the gene
KO perturbed (1 - unperturbed/control leaves; a leaf is unperturbed if the KO kept it
on the manifold at S.OFF_RATIO and snapped it to the same cell as control).  the
pooled distribution is overall recall; coarse anatomical contexts with more than MIN_N
associations are kept as strata.  a matched random-gene null gives the background
perturbed fraction.  writes the per-box median (+ n, mean) for every plotted box to
mgi_box_medians.md.

fast + dataset-free: reads only the ring screen tree pkls for the annotated + random
genes plus the controls.  the tree-readout helpers come from readout.py; the
association set (MP edges + phenotype map) and coarse strata are loaded here.
"""

import os
import pickle

import numpy as np
import pandas as pd

from analysis.figures.perturb import style as S
from analysis.pipelines.perturb.perturb_result import off_manifold_descendants
from analysis.figures.perturb.readout import (
    name_to_celltype, readout_scores, counts_at_ratio,
)

MIN_N = 100           # keep a coarse context with more than this many associations
MIN_CTRL = 50         # need more than this many control cells in a cell type
N_RAND_GENES = 40     # random (non-annotated) genes loaded for the null
N_RAND_PAIRS = 10000  # monte-carlo (gene, cell type) draws for the null

# curated MGI phenotype -> cell type edges + the phenotype->cell-type map.
MP_EDGES = os.path.join(os.path.dirname(S.DATA_DIR), "mouse_phenotypes_gene_attribute_edges.txt")
PHENO_MAP = os.path.join(S.base_dir, "literature", "phenotype_to_celltype_mapping.csv")

# phenotype-map concept keys grouped into coarse anatomical contexts (plot order).
COARSE = [
    ("CNS / brain",    ["brain", "forebrain", "midbrain", "hindbrain", "telencephalon",
                        "pallium", "midbrain-hindbrain boundary"]),
    ("eye",            ["eye", "lens", "lens vesicle", "retina"]),
    ("ear",            ["ear", "inner ear", "otic vesicle"]),
    ("craniofacial",   ["craniofacial", "face", "pharyngeal arch"]),
    ("heart",          ["heart"]),
    ("vascular",       ["vascular", "vascular endothelial cell", "vascular remodeling",
                        "artery", "vein"]),
    ("somite",         ["somite", "dermomyotome", "myotome"]),
    ("gut",            ["digestive system", "intestine"]),
    ("liver",          ["liver", "hepatic", "hepatobiliary system"]),
    ("extraembryonic", ["extraembryonic mesoderm", "allantois"]),
]
CONCEPT_TO_COARSE = {c: name for name, concepts in COARSE for c in concepts}
COARSE_ORDER = [name for name, _ in COARSE]


def load_associations(screened_upper):
    """unique (gene_upper, cell_type, coarse) MGI associations among screened genes."""
    mp = pd.read_csv(MP_EDGES, sep="\t", index_col=0)
    mp = mp[mp["Gene"].str.upper().isin(screened_upper)]
    pm = pd.read_csv(PHENO_MAP).dropna(subset=["cell_type"])
    pm = pm[pm["concept_key"].isin(CONCEPT_TO_COARSE)].copy()
    pm["coarse"] = pm["concept_key"].map(CONCEPT_TO_COARSE)
    edges = mp.merge(pm[["phenotype", "cell_type", "coarse"]],
                     left_on="Phenotype", right_on="phenotype", how="inner")
    edges["Gene"] = edges["Gene"].str.upper()
    return edges[["Gene", "cell_type", "coarse"]].drop_duplicates()


def counts_for_trees(trees, name2ct):
    readouts = []
    for tree in trees:
        readouts += readout_scores(tree, name2ct, S.RING_TARGET_TIME, S.RING_TOL)
    return counts_at_ratio(readouts, S.OFF_RATIO)


def unperturbed_counts(path, ctrl_trees, name2ct):
    """per cell type, the control readout leaves this KO left unperturbed: on the
    manifold at S.OFF_RATIO and snapped to the same cell as in control."""
    counts = {}
    for name, d in pickle.load(open(path, "rb")).items():
        ko, ctrl = d["tree"], ctrl_trees[name]
        bad = off_manifold_descendants(ko, S.OFF_RATIO)
        for n, a in ctrl.nodes(data=True):
            t = a.get("time")
            if (t is not None and abs(t - S.RING_TARGET_TIME) < S.RING_TOL and n not in bad
                    and ko.nodes[n].get("snapped_name") == a.get("snapped_name")):
                ct = name2ct.get(a.get("snapped_name"))
                counts[ct] = counts.get(ct, 0) + 1
    return counts


def main():
    name2ct = name_to_celltype()
    genes_dir = os.path.join(S.RING_SCREEN, "genes")
    gene_by_upper = {os.path.splitext(e.name)[0].upper(): os.path.splitext(e.name)[0]
                     for e in os.scandir(genes_dir) if e.name.endswith(".pkl")}

    assoc = load_associations(set(gene_by_upper))
    assoc = assoc[assoc["Gene"].isin(gene_by_upper)]
    print(f"{len(assoc)} associations over {assoc['Gene'].nunique()} genes", flush=True)

    control, ctrl_trees = {}, {}
    for p in sorted(os.scandir(os.path.join(S.RING_SCREEN, "control")), key=lambda e: e.name):
        tree = ctrl_trees[os.path.splitext(p.name)[0]] = pickle.load(open(p.path, "rb"))["tree"]
        for ct, c in counts_for_trees([tree], name2ct).items():
            control[ct] = control.get(ct, 0) + c

    pert = {gu: unperturbed_counts(os.path.join(genes_dir, f"{gene_by_upper[gu]}.pkl"),
                                   ctrl_trees, name2ct)
            for gu in sorted(assoc["Gene"].unique())}

    rows = []
    for gu, ct, coarse in assoc.itertuples(index=False):
        if control.get(ct, 0) <= MIN_CTRL:
            continue
        rows.append({"gene": gu, "cell_type": ct, "coarse": coarse,
                     "perturbed_fraction": 1.0 - pert[gu].get(ct, 0) / control[ct]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(S.NUMERICAL, "mgi_box.csv"), index=False)

    # null: monte-carlo (gene, cell type) pairs matched to the plotted associations'
    # cell-type distribution but drawn from random non-annotated genes.
    rng = np.random.default_rng(S.seed)
    pool = sorted(set(gene_by_upper) - set(assoc["Gene"]))
    rand_genes = rng.choice(pool, min(N_RAND_GENES, len(pool)), replace=False)
    rand_counts = {gu: unperturbed_counts(os.path.join(genes_dir, f"{gene_by_upper[gu]}.pkl"),
                                          ctrl_trees, name2ct)
                   for gu in rand_genes}
    draw_ct = rng.choice(df["cell_type"].to_numpy(), N_RAND_PAIRS)
    draw_g = rng.choice(list(rand_counts), N_RAND_PAIRS)
    rand_dep = np.array([1.0 - rand_counts[g].get(ct, 0) / control[ct]
                         for g, ct in zip(draw_g, draw_ct)])
    np.savez(os.path.join(S.NUMERICAL, "mgi_box.npz"), rand_dep=rand_dep)

    # plotted boxes: random + all references, then coarse contexts (n>MIN_N) by
    # descending median (the panel-b order).
    coarse_kept = [c for c in COARSE_ORDER if (df["coarse"] == c).sum() > MIN_N]
    frac = lambda c: df.loc[df["coarse"] == c, "perturbed_fraction"].to_numpy()
    coarse_kept.sort(key=lambda c: np.median(frac(c)), reverse=True)
    box_vals = {"random pairs": rand_dep, "all annotated": df["perturbed_fraction"].to_numpy(),
                **{c: frac(c) for c in coarse_kept}}

    lines = [
        "# mgi perturbed-fraction box medians (panel b)", "",
        f"perturbed fraction (1 - unperturbed / control readout leaves) at off-manifold "
        f"ratio {S.OFF_RATIO:g}; coarse anatomical contexts kept at n>{MIN_N}, ordered "
        f"by descending median.", "",
        "| group | n | median | mean |", "| --- | --- | --- | --- |",
    ]
    for name, v in box_vals.items():
        lines.append(f"| {name} | {len(v)} | {np.median(v):.4f} | {v.mean():.4f} |")
    lines.append("")
    out_md = os.path.join(S.NUMERICAL, "mgi_box_medians.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out_md}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
