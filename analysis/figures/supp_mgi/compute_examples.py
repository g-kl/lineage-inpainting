"""panel b's examples, MGI pairs the model depletes where the gene is barely expressed
at E9.0 (reads numerical_results/pairs.csv) -> examples.csv, examples.md.

the pairs are in the knockout's lowest third of its own E9.0 expression (within-gene
percentile ranks), from the TF list, the knockout fully depleting at most MAX_FULL cell
types, depletion at least MIN_DEPLETION. the biggest rank gain from expression to model
first, one per gene. ranks are among the knockout's cell types, 1 = most expressed /
most depleted, ties averaged.

    python -m analysis.figures.supp_mgi.compute_examples
"""
import os

import pandas as pd

from analysis.figures.supp_mgi import style as S


def main():
    d = pd.read_csv(os.path.join(S.NR, "pairs.csv"))
    with open(S.TF_LIST) as fh:
        tf = {ln.strip() for ln in fh if ln.strip()}
    by = d.groupby("gene")
    d = d.assign(expr_rank=by["expression"].rank(ascending=False),
                 model_rank=by["depletion"].rank(ascending=False),
                 low=by["expression"].rank(pct=True) <= 1 / 3,
                 n_full=by["depletion"].transform(lambda s: int((s >= 1).sum())))
    d["gain"] = d["expr_rank"] - d["model_rank"]
    e = d[d["is_mgi"] & d["low"] & d["gene"].isin(tf)
          & (d["n_full"] <= S.MAX_FULL) & (d["depletion"] >= S.MIN_DEPLETION)]
    e = e.sort_values(["gain", "model_rank", "gene"], ascending=[False, True, True])
    ex = e.drop_duplicates("gene").head(S.N_EXAMPLES)[
        ["gene", "cell_type", "coarse", "expression", "expr_rank", "depletion",
         "model_rank", "gain", "n_full"]]
    ex.to_csv(os.path.join(S.NR, "examples.csv"), index=False)

    lines = [
        "# examples (panel b)", "",
        f"{len(e)} MGI pairs pass the rule, from {e['gene'].nunique()} knockouts. ranks "
        f"among the knockout's {d['cell_type'].nunique()} cell types, 1 = most expressed / "
        "most depleted.", "",
        "| gene | cell type | context | expression | rank | depletion | rank |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *(f"| {r.gene} | {r.cell_type} | {r.coarse} | {r.expression:.4f} | "
          f"{r.expr_rank:g} | {r.depletion:.3f} | {r.model_rank:g} |"
          for r in ex.itertuples()), "",
    ]
    out = os.path.join(S.NR, "examples.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
