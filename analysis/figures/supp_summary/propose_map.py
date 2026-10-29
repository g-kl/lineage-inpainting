"""propose celltype_map.csv, our cell type names -> qiu's, for hand curation. see concept.md.
only needs rerunning if the vocabulary changes; `qiu_types` and `status` survive a rerun.

    python -m analysis.figures.supp_summary.propose_map
"""
import collections
import difflib
import os
import pickle
import re

import pandas as pd

from analysis.figures.supp_summary import style as S

SNAP_WINDOW = 1    # the assignments snap a node within this many time bins of its own
DT = 0.5           # the engine's time step
N_CANDIDATES = 4   # difflib suggestions per type
CURATED = ["qiu_types", "status"]
COLS = ["cell_type", "germ_layer", "qiu_types", "candidates", "status"]


def norm(s):
    """lowercase, no parenthetical, no trailing "cell(s)", singular."""
    s = re.sub(r"\s*\(.*?\)\s*", " ", s.lower().strip()).strip()
    s = re.sub(r"\bcells?$", "", s).strip()
    return re.sub(r"s$", "", s)


def vocabulary(labels):
    """cell_type -> germ_layer over every label a node can carry: the label, at its
    time's resolution, of any cell within SNAP_WINDOW bins of it."""
    levels = set(S.RESOLUTION_BY_TIME.values())
    by_stage = collections.defaultdict(lambda: collections.defaultdict(set))
    for name, v in labels.items():
        stage = float(re.match(r"E(\d+(?:\.\d+)?)-", name).group(1))
        for lev in levels:
            by_stage[stage][lev].add(v[lev])
    found = {lab for t, lev in S.RESOLUTION_BY_TIME.items()
             for stage, ls in by_stage.items()
             if abs(stage - t) <= SNAP_WINDOW * DT + 1e-9 for lab in ls[lev]}
    gl = {v["cell_type"]: v["germ_layer"] for v in labels.values()}
    for v in labels.values():
        gl.setdefault(v["lineage"], v["germ_layer"])
        gl.setdefault(v["germ_layer"], v["germ_layer"])
    return {ct: gl.get(ct, "") for ct in sorted(found)}


def propose(ours, theirs):
    """one row per type of ours: the exact `norm` match if there is one, and the
    nearest qiu names."""
    by_norm = collections.defaultdict(list)
    for t in theirs:
        by_norm[norm(t)].append(t)
    rows = []
    for ct, layer in ours.items():
        exact = by_norm.get(norm(ct), [])
        near = difflib.get_close_matches(norm(ct), list(by_norm), N_CANDIDATES, 0.6)
        rows.append({"cell_type": ct, "germ_layer": layer,
                     "qiu_types": ";".join(sorted(exact)),
                     "candidates": ";".join(t for k in near for t in sorted(by_norm[k])),
                     "status": "auto" if exact else "todo"})
    return pd.DataFrame(rows, columns=COLS)


def merge(new, path):
    """curated columns win wherever the existing csv has a row for the type."""
    if not os.path.exists(path):
        return new
    old = pd.read_csv(path, dtype=str).fillna("").set_index("cell_type")
    for i, r in new.iterrows():
        if r.cell_type in old.index:
            new.loc[i, CURATED] = old.loc[r.cell_type, CURATED].values
    return new


def main():
    with open(S.MAP_PKL, "rb") as f:
        voc = vocabulary(pickle.load(f)["name_to_labels"])
    if os.path.exists(S.MAP_CSV):
        gone = sorted(set(pd.read_csv(S.MAP_CSV, dtype=str).cell_type) - set(voc))
        if gone:
            print(f"dropped, outside the vocabulary: {', '.join(gone)}")
    q = pd.read_csv(S.QIU)
    df = merge(propose(voc, set(q.source) | set(q.target)), S.MAP_CSV)
    df.to_csv(S.MAP_CSV, index=False)

    print(f"wrote {S.MAP_CSV}: {(df.qiu_types != '').sum()} of {len(df)} types mapped")
    print(df.status.value_counts().to_string())
    for r in df[df.status == "todo"].itertuples():
        print(f"  todo {r.cell_type:32s} {r.candidates}")


if __name__ == "__main__":
    main()
