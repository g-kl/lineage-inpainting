"""real-cell context motif -> context_realcells.csv (one row per motif x method).

reads context_landscape/{landscape,samples}.csv. keeps the canonical method order plus
the oracle / marg reference rows and joins each motif's drawn cell types back on.

    python -m synthetic_evals.figure.compute_context_realcells
"""
import os

import pandas as pd

from synthetic_evals.figure import style as S

SPLITS = ["", "_ambiguous", "_unambiguous"]


def main():
    land = pd.read_csv(os.path.join(S.CONTEXT_LANDSCAPE_DIR, "landscape.csv"))
    samples = pd.read_csv(os.path.join(S.CONTEXT_LANDSCAPE_DIR, "samples.csv"))

    present = set(land["method"])
    order = ([m for m in S.METHODS if m in present]
             + [m for m in S.ABLATIONS if m in present]
             + [m for m in S.REFERENCES if m in present])
    land = land[land["method"].isin(order)].copy()
    land["method"] = pd.Categorical(land["method"], categories=order, ordered=True)

    meta = samples.set_index("cfg")[["root_type", "A", "C", "B"]]
    out = (land.drop(columns=["root_type", "A", "C", "B"])
               .join(meta, on="cfg")
               .sort_values(["seed", "method"]))
    cols = (["cfg", "seed", "method"]
            + [f"mse{s}" for s in SPLITS] + [f"type_acc{s}" for s in SPLITS]
            + ["root_type", "A", "C", "B"])
    out[cols].to_csv(os.path.join(S.NR, "context_realcells.csv"), index=False)

    acc = out.loc[out["method"] == "ours", "type_acc_ambiguous"]
    print(f"wrote context real-cell cache -> {S.NR}  "
          f"({out['cfg'].nunique()} motifs, methods={order})")
    print(f"  ours type_acc on {{B,B}} progenitors: mean {acc.mean():.3f}  "
          f"min {acc.min():.3f}  over {len(acc)} motifs")


if __name__ == "__main__":
    main()
