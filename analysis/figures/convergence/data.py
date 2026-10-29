"""loaders shared by rows d-g: the label maps, the trajectory cache compute_paths.py
writes, the h5ad annotation and expression, and the ~E8.5 source of every trajectory."""
import os
import sys

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as S

NODE = ("panel", "tree", "traj", "group", "time", "row", "is_leaf")
GAP_NNZ = 20_000   # csr rows this far apart are read as separate spans


def labels():
    """cell_subtype (lowercased) -> cell_type, and cell_type -> lineage.

    extraembryonic rows are dropped before the lineage map is built: 'endothelial
    cell' is listed under both mesoderm and its yolk sac lineage, and without this it
    resolves to the yolk sac one and disappears from mesoderm entirely."""
    ti = pd.read_csv(S.TYPE_INFO)
    sub2type = {k.lower(): v for k, v in zip(ti.cell_subtype, ti.cell_type)}
    ti = ti[~ti.lineage.str.startswith("extraembryonic")]
    return sub2type, dict(zip(ti.cell_type, ti.lineage))


def germ_layers():
    ti = pd.read_csv(S.TYPE_INFO).drop_duplicates("cell_type")
    return dict(zip(ti.cell_type, ti.germ_layer))


def read_rows(rows, ncol):
    """dense expression for sorted `rows`, reading the csr in contiguous spans."""
    out = np.zeros((len(rows), ncol), np.float32)
    with h5py.File(S.H5AD, "r") as f:
        X = f["X"]
        ptr, data, ind = X["indptr"][:], X["data"], X["indices"]
        i = 0
        while i < len(rows):
            j = i
            while (j + 1 < len(rows)
                   and ptr[rows[j + 1]] - ptr[rows[j] + 1] < GAP_NNZ):
                j += 1
            s, e = int(ptr[rows[i]]), int(ptr[rows[j] + 1])
            d, k = data[s:e], ind[s:e]
            for m in range(i, j + 1):
                a, b = int(ptr[rows[m]]) - s, int(ptr[rows[m] + 1]) - s
                out[m, k[a:b]] = d[a:b]
            i = j + 1
    return out


def nodes():
    """every traced node with its h5ad row and latent, as compute_paths.py cached it.
    only trajectories that reach a program's target are in there."""
    z = np.load(os.path.join(S.NR, "cache_paths.npz"), allow_pickle=True)
    return pd.DataFrame({c: z[c] for c in NODE}), z["Z"]


def axes():
    return np.load(os.path.join(S.NR, "axes.npz"))


def obs():
    """cell names and cell types for every row of the h5ad."""
    sub2type, _ = labels()
    with h5py.File(S.H5AD, "r") as f:
        o = f["obs"]
        name = np.array([n.decode() for n in o["_index"][:]])
        sub = pd.Categorical.from_codes(
            o["cell_subtype"]["codes"][:],
            [c.decode() for c in o["cell_subtype"]["categories"][:]]).astype(str)
    return name, pd.Series(sub).str.lower().map(sub2type).fillna("").values


def sheet_table(d, Z, sheet, name, ctype, V, mu):
    """one row per trajectory: the assigned ancestor nearest T_SRC and the E9.5 leaf
    it reaches, both with umap coordinates and both scored on the two axes.

    two kinds of anchor are dropped, and `funnel` counts both: the wrong germ layer,
    since a mesodermal fate reached through an ectodermal ancestor is a coupling
    artifact and not a source, and anchors already at the destination, which are not
    a source of anything either."""
    m = d.panel.isin(sheet["panels"])
    s = d[m & (d.is_leaf == 0) & (abs(d.time - S.T_SRC) < S.SRC_TOL)].copy()
    s["off"] = (s.time - S.T_SRC).abs()
    s = s.sort_values("off").drop_duplicates(["panel", "traj"])
    lf = d[m & (d.is_leaf == 1)]
    funnel = dict(traj=int(d[m].traj.nunique()), anchored=len(s))

    gl = germ_layers()
    st = ctype[s.row.values]
    ok = np.array([gl.get(a, "") == sheet["germ"] for a in st])
    funnel["wrong_germ"] = int((~ok).sum())
    s, st = s[ok], st[ok]
    ok = ~np.isin(st, list(sheet["at_dest"]))
    funnel["at_dest"] = int((~ok).sum())
    s, st = s[ok], st[ok]
    funnel["kept"] = len(s)

    _, type2lin = labels()
    xcls = S.PROGRAMS[sheet["axes"]]["xcls"]
    Sc = (Z[s.index.values] - mu) @ V
    src = pd.DataFrame(dict(
        panel=s.panel.values, traj=s.traj.values, tree=s.tree.values,
        src_cell=name[s.row.values], src_row=s.row.values, src_type=st,
        bin=[xcls(a, type2lin.get(a, "")) for a in st], src_time=s.time.values,
        x=Sc[:, 0], y=Sc[:, 1]))
    L = (Z[lf.index.values] - mu) @ V
    leaf = pd.DataFrame(dict(panel=lf.panel.values, traj=lf.traj.values,
                             leaf_cell=name[lf.row.values], leaf_row=lf.row.values,
                             x_leaf=L[:, 0], y_leaf=L[:, 1]))
    out = src.merge(leaf, on=["panel", "traj"], how="inner")
    out["target"] = out.panel.map({p: S.PROGRAMS[p]["target"] for p in sheet["panels"]})
    return out, funnel


def umap():
    return pd.read_csv(S.UMAP_CSV, index_col=0)


def background():
    """the same 120k-cell backdrop the row-a umap panels use."""
    return pd.read_pickle(os.path.join(S.NR, "umap_bg.pkl"))
