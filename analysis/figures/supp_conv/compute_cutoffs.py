"""panel d: fig 4 panel f's erased share over a grid of the two cutoffs. a gene is
divergent if its E8.5 |log2 fc| reaches xcut, and erased if its E9.5 |log2 fc| then
stays under ycut -- the inequalities of `compute_divergence.coords`.

    python -m analysis.figures.supp_conv.compute_cutoffs
"""
import os

import pandas as pd

from analysis.figures.supp_conv import style as S


def main():
    G = pd.read_csv(os.path.join(S.C.NR, "gene_divergence.csv"))
    rows = []
    for panel, g in G.groupby("panel"):
        for xc in S.XCUTS:
            for yc in S.YCUTS:
                div = g.x >= xc
                rows.append(dict(panel=panel, xcut=xc, ycut=yc, n_div=int(div.sum()),
                                 n_erased=int((div & (g.y < yc)).sum())))
    D = pd.DataFrame(rows)
    D["erased"] = D.n_erased / D.n_div

    # the fig 4 cell reproduces panel f's classes
    at = D[(D.xcut == S.C.DIV_XCUT) & (D.ycut == S.C.DIV_YCUT)].set_index("panel")
    n = G.groupby(["panel", "cls"]).size().unstack(fill_value=0)
    assert (at.n_erased == n.erased).all(), (at, n)
    assert (at.n_div - at.n_erased == n.memory).all(), (at, n)

    top = D.loc[D.groupby(["xcut", "ycut"]).erased.idxmax()].panel
    print(f"most erased fate over the {len(top)}-cell grid: "
          f"{top.value_counts().to_dict()}")
    print("at fig 4's cutoffs: " + ", ".join(
        f"{r.Index} {r.erased:.2f} ({r.n_erased}/{r.n_div})" for r in at.itertuples()))
    D.to_csv(os.path.join(S.NR, "cutoffs.csv"), index=False)
    print(f"wrote cutoffs.csv to {S.NR}")


if __name__ == "__main__":
    main()
