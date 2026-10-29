"""compute the gnomad LoF-constraint box panel -> numerical_results/gnomad_box.csv.

bins the E9.0 HVG ring screen genes by gnomad LOEUF -- the N_BIN most constrained
(LoF intolerant) vs the N_BIN least constrained (LoF tolerant) -- and scores each by
effect_norm (l2 norm of the per-cell-type abundance log2fc vector) at S.OFF_RATIO.
writes the per-bin median (+ n, mean) and a mann-whitney p to gnomad_box_medians.md.

fast + dataset-free: reads only the ring screen tree pkls for the 2*N_BIN binned
genes plus the controls.
"""

import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analysis.figures.perturb import style as S
from analysis.figures.perturb.readout import (
    name_to_celltype, readout_scores, counts_at_ratio, effect_norm,
)
from utils.human_to_mouse_gene_names import human_to_mouse

N_BIN = 100                       # genes per constraint bin (most / least constrained)
BINS = ["intolerant", "tolerant"]
GNOMAD_TSV = os.path.join(S.base_dir, "literature", "gnomad_constraints.tsv")


def load_loeuf():
    """mouse gene -> LOEUF (min over the human ortholog's canonical coding transcripts)."""
    g = pd.read_csv(GNOMAD_TSV, sep="\t",
                    usecols=["gene", "canonical", "transcript_type", "lof.oe_ci.upper"],
                    low_memory=False)
    g = g[(g["canonical"] == True) & (g["transcript_type"] == "protein_coding")]
    g = g.dropna(subset=["gene", "lof.oe_ci.upper"])
    human_loeuf = g.groupby("gene")["lof.oe_ci.upper"].min().to_dict()
    h2m = human_to_mouse(list(human_loeuf))
    return {m: human_loeuf[h] for h, m in h2m.items()}


def pooled_counts(paths, name2ct):
    readouts = []
    for p in paths:
        obj = pickle.load(open(p, "rb"))
        trees = [obj["tree"]] if "tree" in obj else [d["tree"] for d in obj.values()]
        for tree in trees:
            readouts += readout_scores(tree, name2ct, S.RING_TARGET_TIME, S.RING_TOL)
    return counts_at_ratio(readouts, S.OFF_RATIO)


def main():
    name2ct = name_to_celltype()
    mouse_loeuf = load_loeuf()

    genes_dir = os.path.join(S.RING_SCREEN, "genes")
    have = {os.path.splitext(e.name)[0] for e in os.scandir(genes_dir)
            if e.name.endswith(".pkl")}
    scored = sorted((g for g in mouse_loeuf if g in have), key=lambda g: mouse_loeuf[g])
    k = min(N_BIN, len(scored) // 2)
    mouse_bins = {"intolerant": scored[:k], "tolerant": scored[-k:]}
    print(f"{len(scored)} genes with LOEUF + pkl; {k} per bin", flush=True)

    ctrl = pooled_counts([e.path for e in os.scandir(os.path.join(S.RING_SCREEN, "control"))],
                         name2ct)

    scores = {}
    for name in BINS:
        for gene in mouse_bins[name]:
            counts = pooled_counts([os.path.join(genes_dir, f"{gene}.pkl")], name2ct)
            scores[gene] = effect_norm(counts, ctrl)

    rows = [{"bin": name, "gene": g, "loeuf": mouse_loeuf[g], "effect_norm": scores[g]}
            for name in BINS for g in mouse_bins[name]]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(S.NUMERICAL, "gnomad_box.csv"), index=False)

    vals = {name: df.loc[df["bin"] == name, "effect_norm"].dropna().to_numpy() for name in BINS}
    u, pval = mannwhitneyu(vals["intolerant"], vals["tolerant"])
    labels = {"intolerant": "LoF intolerant", "tolerant": "LoF tolerant"}
    lines = [
        "# gnomad box medians (panel c)", "",
        f"perturbation effect (abundance log2fc L2 norm) at off-manifold ratio "
        f"{S.OFF_RATIO:g}, for the {k} most vs {k} least LoF-constrained ring-screen "
        f"genes (gnomad LOEUF).", "",
        "| bin | n | median | mean |", "| --- | --- | --- | --- |",
    ]
    for name in BINS:
        v = vals[name]
        lines.append(f"| {labels[name]} | {len(v)} | {np.median(v):.4f} | {v.mean():.4f} |")
    lines += ["", f"mann-whitney u={u:.0f}, p={pval:.3g}", ""]
    out_md = os.path.join(S.NUMERICAL, "gnomad_box_medians.md")
    with open(out_md, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {out_md}", flush=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, S.base_dir)
    os.chdir(S.base_dir)
    main()
