"""Visualise the GraftExtender on real mouse-embryo data.

Renders, side by side:
  1. A "donor" tree from the latest record (the leaf_time = t_late tree).
  2. An early tree before extension.
  3. The same early tree after extension.

For each tree we print structural stats: node count, leaf count, depth,
mean out-degree at interior nodes, time range.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from datasets.tree_record import TreeRecord   # noqa: E402
from extenders import SubtreeBank   # noqa: E402
from extenders.graft import GraftExtender   # noqa: E402


# ----- layout -------------------------------------------------------------

def fmt_stats(name, values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return f"  {name}: empty"
    return (f"  {name}: n={len(arr)} min={np.nanmin(arr):.3f} "
            f"max={np.nanmax(arr):.3f} mean={np.nanmean(arr):.3f}")


def tree_layout(g, root):
    """Return {node: (x, y)} with y = time, x spread by descendant count.

    Iterative two-pass (post-order) layout so it scales past Python recursion.
    """
    times = nx.get_node_attributes(g, "time")
    children = {n: list(g.successors(n)) for n in g.nodes()}

    # iterative DFS for leaf order
    order = []
    stack = [(root, iter(children[root]))]
    visited = {root}
    leaf_x = {}
    # Iterative post-order via explicit stack.
    work = [(root, False)]
    while work:
        n, processed = work.pop()
        if processed:
            if not children[n]:
                leaf_x[n] = len(order)
                order.append(n)
            continue
        work.append((n, True))
        for c in reversed(children[n]):
            work.append((c, False))

    pos = {}
    # iterative post-order to compute x = mean(child x).
    work = [(root, False)]
    while work:
        n, processed = work.pop()
        if processed:
            if not children[n]:
                pos[n] = (float(leaf_x[n]), times.get(n, 0.0))
            else:
                xs = [pos[c][0] for c in children[n]]
                pos[n] = (float(np.mean(xs)), times.get(n, 0.0))
            continue
        work.append((n, True))
        for c in children[n]:
            work.append((c, False))
    return pos


def draw_tree(ax, g, root, *, observed_set, added_set, title):
    pos = tree_layout(g, root)
    missing = [n for n in g if n not in pos]
    if missing:
        print(f"  WARN: {len(missing)} nodes unreachable from root {root!r}; "
              f"first: {missing[:3]}")

    # Edges
    for u, v in g.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        added_edge = (u in added_set) or (v in added_set)
        color = "#bbbbbb" if added_edge else "#444444"
        lw = 0.8 if added_edge else 1.2
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, zorder=1)

    # Nodes
    xs_obs, ys_obs = [], []
    xs_int, ys_int = [], []
    xs_added, ys_added = [], []
    for n, (x, y) in pos.items():
        if n in added_set:
            xs_added.append(x); ys_added.append(y)
        elif n in observed_set:
            xs_obs.append(x); ys_obs.append(y)
        else:
            xs_int.append(x); ys_int.append(y)
    if xs_added:
        ax.scatter(xs_added, ys_added, s=14, c="#aaaaaa", zorder=2,
                   edgecolors="none", label=f"grafted ({len(xs_added)})")
    if xs_obs:
        ax.scatter(xs_obs, ys_obs, s=22, c="#d6543a", zorder=3,
                   edgecolors="none", label=f"observed ({len(xs_obs)})")
    if xs_int:
        ax.scatter(xs_int, ys_int, s=14, c="#3c4ebf", zorder=2,
                   edgecolors="none", label=f"internal ({len(xs_int)})")

    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()
    ax.set_ylabel("time")
    ax.set_xticks([])
    ax.legend(loc="lower right", fontsize=7, frameon=False)


# ----- stats --------------------------------------------------------------

def tree_stats(g, root):
    times = nx.get_node_attributes(g, "time")
    leaves = [n for n in g if g.out_degree(n) == 0]
    interior = [n for n in g if g.out_degree(n) > 0]
    out_degs = [g.out_degree(n) for n in interior]
    depths = nx.shortest_path_length(g, source=root)
    return {
        "n_nodes": g.number_of_nodes(),
        "n_leaves": len(leaves),
        "n_interior": len(interior),
        "max_depth": max(depths.values()) if depths else 0,
        "mean_branching": float(np.mean(out_degs)) if out_degs else 0.0,
        "max_branching": int(max(out_degs)) if out_degs else 0,
        "time_min": min(times.values()) if times else float("nan"),
        "time_max": max(times.values()) if times else float("nan"),
    }


def print_stats(name, s):
    print(
        f"  {name:24s} nodes={s['n_nodes']:4d} leaves={s['n_leaves']:4d} "
        f"interior={s['n_interior']:4d} depth={s['max_depth']:3d} "
        f"mean_brn={s['mean_branching']:.2f} max_brn={s['max_branching']:2d} "
        f"t=[{s['time_min']:.2f},{s['time_max']:.2f}]"
    )


# ----- main ---------------------------------------------------------------

def _list_cache_by_time():
    cache_dir = "cache"
    out = []
    for fn in os.listdir(cache_dir):
        if not fn.endswith(".pkl"):
            continue
        try:
            tag = fn.split("_E", 1)[1]
            t = float(tag.split("-", 1)[0])
        except Exception:
            continue
        out.append((t, os.path.join(cache_dir, fn)))
    out.sort()
    if not out:
        raise FileNotFoundError("no cache files in cache/")
    return out


def _pick_early_late_cache():
    """Adjacent-record pick to match PairedOffsetSampler(target_gap~0.5).

    Env: EARLY_T (default 8.5), LATE_T (default 9.0).
    """
    target_early = float(os.environ.get("EARLY_T", "8.5"))
    target_late  = float(os.environ.get("LATE_T", "9.0"))
    all_caches = _list_cache_by_time()
    early = min((c for c in all_caches if c[0] <= target_early),
                key=lambda c: abs(c[0] - target_early), default=all_caches[0])
    late  = min((c for c in all_caches if c[0] >= target_late),
                key=lambda c: abs(c[0] - target_late), default=all_caches[-1])
    return early, late


def main():
    (t_early, early_path), (t_late, late_path) = _pick_early_late_cache()
    print(f"early cache: {early_path}  t={t_early}", flush=True)
    print(f"late  cache: {late_path}  t={t_late}", flush=True)

    early = TreeRecord.load(early_path)
    late = TreeRecord.load(late_path)
    early_name = os.path.basename(early_path).split("_E", 1)[1].rsplit(".pkl", 1)[0]
    late_name  = os.path.basename(late_path).split("_E", 1)[1].rsplit(".pkl", 1)[0]
    t_early = float(early.leaf_time)
    t_late = float(late.leaf_time)
    print(f"early rec: {early_name}  leaf_time={t_early}")
    print(f"late  rec: {late_name}  leaf_time={t_late}")

    rng = np.random.default_rng(0)

    # Early subtrees: pick a deeper root_time so the original has meaningful
    # structure (not just a parent + 2 leaves). Env: EARLY_ROOT_T overrides.
    early_root_t = float(os.environ.get("EARLY_ROOT_T", str(t_early - 1.5)))
    pool = early.get_subtrees(time=early_root_t, n=64, rng=rng)
    pool = [s for s in pool if s["size"] >= 4]   # drop trivial 2-leaf subtrees
    if not pool:
        pool = early.get_subtrees(time=early_root_t, n=8, rng=rng)
    rng.shuffle(pool)
    early_subtrees = pool[:8]
    assert early_subtrees, "no early subtrees available"
    print(f"early subtrees: {len(early_subtrees)} "
          f"(root_time={early_subtrees[0]['root_time']:.2f}, "
          f"leaf_time={early_subtrees[0]['leaf_time']:.2f}, "
          f"sizes={[s['size'] for s in early_subtrees]})")

    # Donor pool: the SubtreeBank's (t_early, t_late) stratum -- the late
    # record's trees layer-sliced at the early leaf time. This is exactly what
    # the extender grafts (its SubtreeBank).
    extender = GraftExtender()
    extender.bank = SubtreeBank({early_name: early, late_name: late})
    stratum = extender.bank.strata.get((float(t_early), float(t_late)))
    donors = stratum.donors if stratum is not None else []
    print(f"donor stratum ({t_early}->{t_late}): {len(donors)} candidates")
    if donors:
        print(fmt_stats("donor sizes (leaves)", [d.size for d in donors]))
        print(fmt_stats("donor node counts",   [len(d.nodes) for d in donors]))

    extended = extender(early_subtrees, target_leaf_time=t_late, rng=rng)

    # Pick a representative donor (median leaf count), materialized for viz.
    donor_idx = int(np.argsort([d.size for d in donors])[len(donors) // 2])
    _donor_root, _donor_graph = donors[donor_idx].materialize()
    donor_pick = {
        "graph": _donor_graph, "root": _donor_root,
        "size": donors[donor_idx].size, "leaf_time": t_late,
    }

    # ----- structural stats ------
    print("\nstructural stats:")
    print_stats(f"donor ({late_name})",     tree_stats(donor_pick["graph"], donor_pick["root"]))
    for i in range(min(2, len(early_subtrees))):
        print_stats(f"orig early #{i}",      tree_stats(early_subtrees[i]["graph"], early_subtrees[i]["root"]))
        print_stats(f"extended early #{i}",  tree_stats(extended[i]["graph"], extended[i]["root"]))

    # Aggregate over the batch.
    o_stats = [tree_stats(t["graph"], t["root"]) for t in early_subtrees]
    e_stats = [tree_stats(t["graph"], t["root"]) for t in extended]

    # Exchangeability reference: real late-record subtrees rooted near the
    # early subtrees' own root_time, with the same node-count cap we apply
    # to grafted donors. If extended trees are "exchangeable" with real
    # late trees they should land in the same stat distribution.
    early_root_t = float(np.median([t["root_time"] for t in early_subtrees]))
    real_late_pool = late.get_subtrees(time=early_root_t, n=64, rng=rng)
    real_late_pool = [
        t for t in real_late_pool
        if t["graph"].number_of_nodes() <= 256
    ]
    r_stats = [tree_stats(t["graph"], t["root"]) for t in real_late_pool]

    print(f"\nreal late-rec subtrees for comparison "
          f"(rooted near t={early_root_t:.2f}): n={len(r_stats)}")
    print("\nbatch summaries (mean +/- std):")
    for k in ("n_nodes", "n_leaves", "max_depth", "mean_branching"):
        ov = np.array([s[k] for s in o_stats])
        ev = np.array([s[k] for s in e_stats])
        rv = np.array([s[k] for s in r_stats]) if r_stats else np.array([])
        line = (f"  {k:14s} orig={ov.mean():7.2f}+/-{ov.std():6.2f}   "
                f"ext={ev.mean():7.2f}+/-{ev.std():6.2f}")
        if r_stats:
            line += f"   real_late={rv.mean():7.2f}+/-{rv.std():6.2f}"
        print(line)

    # ----- visualization ------
    # "observed" in the viz sense = real-data node (existed before grafting);
    # "added" = node introduced by the extender (has the ext-prefix).
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=120)

    # All donor nodes are real-data (no extension applied to the donor itself).
    # Mark only the leaves as observed (those are real cells with embeddings);
    # interior nodes are real-data but unobserved.
    donor_g = donor_pick["graph"]
    donor_leaves = {n for n in donor_g if donor_g.out_degree(n) == 0}
    draw_tree(
        axes[0, 0],
        donor_g, donor_pick["root"],
        observed_set=donor_leaves, added_set=set(),
        title=f"donor (slice late at t={t_early:.2f}) "
              f"size={donor_pick['size']} leaves at t={donor_pick['leaf_time']:.2f}",
    )

    for col in (1, 2):
        i = col - 1
        if i >= len(early_subtrees):
            axes[0, col].axis("off")
            continue
        og = early_subtrees[i]["graph"]
        og_leaves = {n for n in og if og.out_degree(n) == 0}
        draw_tree(
            axes[0, col],
            og, early_subtrees[i]["root"],
            observed_set=og_leaves, added_set=set(),
            title=f"orig early #{i} leaves at t={early_subtrees[i]['leaf_time']:.2f}",
        )

    # row 1: 3 extended trees. Observed = original early leaves (still real
    # cells, now internal). Added = nodes minted by the extender.
    for col in range(3):
        if col >= len(extended):
            axes[1, col].axis("off")
            continue
        og = early_subtrees[col]["graph"]
        og_nodes = set(og.nodes())
        og_leaves = {n for n in og if og.out_degree(n) == 0}
        eg = extended[col]["graph"]
        added = set(eg.nodes()) - og_nodes
        draw_tree(
            axes[1, col],
            eg, extended[col]["root"],
            observed_set=og_leaves, added_set=added,
            title=f"extended #{col} target_leaf_time={t_late:.2f}",
        )

    # Backward-edge audit on every extended graph.
    n_bad = 0
    for i, s in enumerate(extended):
        eg = s["graph"]
        bad_edges = []
        for u, v in eg.edges():
            tu = eg.nodes[u].get("time", float("nan"))
            tv = eg.nodes[v].get("time", float("nan"))
            if tv < tu - 1e-6:
                bad_edges.append((u, tu, v, tv))
        if bad_edges:
            n_bad += 1
            print(f"  ext #{i}: {len(bad_edges)} backward edges, e.g. "
                  f"{bad_edges[0][0]}(t={bad_edges[0][1]:.3f})"
                  f"->{bad_edges[0][2]}(t={bad_edges[0][3]:.3f})")
    if n_bad == 0:
        print("\nbackward-edge audit: 0 backward edges across all extended trees.")
    else:
        print(f"\nBACKWARD EDGES FOUND in {n_bad}/{len(extended)} extended trees.")

    out = "outputs_viz/graft_extender_demo.png"
    os.makedirs("outputs_viz", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
