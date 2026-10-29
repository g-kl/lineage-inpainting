#!/usr/bin/env python
"""quantify TF reuse across cell types from a subway_data_<rank>.json.

for each TF, count the number of distinct cell types in which it shows up as a
top hit (across the outer cell-type bins), then plot a histogram of that count.
a tall left bar = mostly cell-type-specific TFs; a long right tail = TFs reused
across many cell types.
"""

import os
import sys
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "subway_data_specificity.json")
GENE_AGE = os.path.join(HERE, "..", "..", "..", "literature", "gene_age.csv")


def cell_types_per_tf(data):
    """tf -> set of cell types it appears in as a top hit (across the outer
    bins). the inner lineage_tfs are excluded since they have no cell-type
    resolution."""
    hits = defaultdict(set)
    for cell in data["cells"]:
        for tfs in cell["tfs"].values():
            for tf in tfs:
                hits[tf].add(cell["cell_type"])
    return hits


def load_gene_age():
    """uppercase gene name -> gene age in My, stripping any leading '>'."""
    df = pd.read_csv(GENE_AGE, usecols=["external_gene_name", "gene_age"])
    age = df["gene_age"].astype(str).str.lstrip(">").astype(float)
    return dict(zip(df["external_gene_name"].str.upper(), age))


def plot_hist(counts, top):
    vals = list(counts.values())
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.hist(vals, bins=range(1, top + 2), align="left",
            color="#4c72b0", edgecolor="white")
    ax.set_xlabel("# cell types a TF is a top hit in")
    ax.set_ylabel("# TFs")
    ax.set_xticks(range(1, top + 1))
    ax.set_title("TF reuse across cell types")
    fig.tight_layout()
    out = os.path.join(HERE, "tf_reuse_cell_type.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def plot_age_by_reuse(counts, top):
    """gene age (log scale) for specific (<3 cell types) vs. reused (>=3) TFs,
    with a Mann-Whitney U test for the difference between the two groups."""
    age = load_gene_age()
    groups = {"<3": [], ">=3": []}
    missing = []
    for tf, c in counts.items():
        a = age.get(tf.upper())
        (groups["<3" if c < 3 else ">=3"].append(a) if a is not None
         else missing.append(tf))
    print(f"{len(missing)} TFs had no gene-age entry"
          + (f": {', '.join(sorted(missing))}" if missing else ""))

    labels = ["<3", ">=3"]
    data = [groups[k] for k in labels]
    stat, p = mannwhitneyu(data[0], data[1], alternative="two-sided")
    meds = [np.median(d) for d in data]
    print(f"medians: <3 = {meds[0]:.0f} My, >=3 = {meds[1]:.0f} My")
    print(f"Mann-Whitney U = {stat:.0f}, p = {p:.3g}")

    pos = [1, 2]
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.boxplot(data, positions=pos, widths=0.6, showfliers=False,
               medianprops=dict(color="#c44e52"))
    for x0, vals in zip(pos, data):
        x = np.random.default_rng(x0).normal(x0, 0.06, len(vals))
        ax.scatter(x, vals, s=14, color="#4c72b0", alpha=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("gene age (My)")
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{k}\n(n={len(v)})" for k, v in zip(labels, data)])
    ax.set_xlabel("# cell types a TF is a top hit in")
    ax.set_title(f"gene age vs. reuse\nMann-Whitney p = {p:.3g}")
    fig.tight_layout()
    out = os.path.join(HERE, "tf_reuse_age.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main():
    data = json.load(open(JSON))
    hits = cell_types_per_tf(data)
    counts = {tf: len(g) for tf, g in hits.items()}
    n_grp = len(set().union(*hits.values()))
    top = max(counts.values())

    print(f"{len(counts)} TFs across {n_grp} cell types; "
          f"{sum(c > 1 for c in counts.values())} reused in >1 cell type")
    for n in (top, top - 1):
        in_bin = sorted(tf for tf, c in counts.items() if c == n)
        print(f"genes in the {n}-cell-type bin: {', '.join(in_bin)}")

    plot_hist(counts, top)
    plot_age_by_reuse(counts, top)


if __name__ == "__main__":
    main()
