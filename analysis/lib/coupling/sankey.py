import math
from collections import defaultdict

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon


PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#8CD17D", "#B6992D", "#499894", "#D37295",
    "#A0CBE8", "#FFBE7D", "#D4A6C8", "#79706E", "#FABFD2",
]


# ---- graph construction ----

def _test_and_filter(edge_counts, row_counts, col_counts, totals,
                     min_support, min_fold, alpha):
    """hypergeometric-style enrichment test with bh correction, returns filtered digraph"""
    stats = []
    for (s, t), x in edge_counts.items():
        m = totals[s[1]]
        r = row_counts[s]
        c = col_counts[t]
        mu = r * c / m

        if m > 1:
            var = mu * (1 - r / m) * (1 - c / m) * m / (m - 1)
        else:
            var = 0.0

        z = 0.0 if var <= 0 else (x - mu) / math.sqrt(var)
        p = 0.5 * math.erfc(z / math.sqrt(2))
        fold = math.inf if mu == 0 else x / mu
        stats.append((s, t, x, mu, fold, z, p))

    # benjamini-hochberg correction
    if len(stats) > 0:
        pvals = np.array([x[6] for x in stats], dtype=float)
        order = np.argsort(pvals)
        qvals = np.empty_like(pvals)
        m = len(pvals)
        best = 1.0
        for k in range(m - 1, -1, -1):
            i = order[k]
            best = min(best, pvals[i] * m / (k + 1))
            qvals[i] = best
    else:
        qvals = np.array([])

    out = nx.DiGraph()
    for rec, q in zip(stats, qvals):
        s, t, x, mu, fold, z, p = rec
        if x >= min_support and fold >= min_fold and q <= alpha:
            out.add_edge(s, t, count=int(x), expected=float(mu),
                         fold=float(fold), z=float(z),
                         p=float(p), q=float(q))

    return out


def build_transition_graph(
    trees,
    name_to_cell_type,
    assignments,
    time_key='time',
    dt=0.5,
    nn_cutoff=None,
    nn_quantile=0.8,
    min_support=10,
    min_fold=2.0,
    alpha=0.05,
    tol=1e-8,
):
    """build enrichment-tested lineage graph from tree assignments.

    nodes are (cell_type, quantized_time) tuples.
    edges carry count, expected, fold, z, p, q attributes.
    """
    if nn_cutoff is None:
        ds = np.fromiter(
            (d for x in assignments['nn_distances'].values() for d in x.values()),
            dtype=float,
        )
        nn_cutoff = np.quantile(ds, nn_quantile)

    edge_counts = defaultdict(int)
    row_counts = defaultdict(int)
    col_counts = defaultdict(int)
    totals = defaultdict(int)
    observed_counts = defaultdict(int)

    def qtime(x):
        return max(7.5, round(round(x / dt) * dt, 6))

    # observed marginals are defined by leaf nodes only and do not depend on
    # assignment quality, so node sizes stay fixed across methods.
    for _, g in trees.items():
        for n in g.nodes():
            if g.out_degree(n) != 0:
                continue
            if n not in name_to_cell_type:
                continue
            tval = g.nodes[n].get(time_key)
            if tval is None:
                continue
            observed_counts[(name_to_cell_type[n], qtime(tval))] += 1

    for tree_name, g in trees.items():
        amap = assignments[tree_name]
        dmap = assignments['nn_distances'][tree_name]

        roots = [n for n in g if g.in_degree(n) == 0]
        stack = [(r, None) for r in roots]

        while stack:
            n, last = stack.pop()
            cur = None

            if g.out_degree(n) == 0:
                cur = (name_to_cell_type[n], qtime(g.nodes[n][time_key]))
            else:
                obs = amap.get(n)
                if obs is not None and dmap[n] <= nn_cutoff:
                    if obs not in name_to_cell_type:
                        continue
                    cur = (name_to_cell_type[obs], qtime(g.nodes[n][time_key]))

            if cur is not None:
                if last is not None:
                    if abs(cur[1] - last[1] - dt) < tol:
                        s = last
                        t = cur
                        edge_counts[(s, t)] += 1
                        row_counts[s] += 1
                        col_counts[t] += 1
                        totals[s[1]] += 1
                last = cur

            for c in g.successors(n):
                stack.append((c, last))

    out = _test_and_filter(edge_counts, row_counts, col_counts, totals,
                           min_support, min_fold, alpha)
    for node, count in observed_counts.items():
        out.add_node(node, observed_count=int(count))
    return out


def build_transition_graph_genot(
    ancestor_map,
    name_to_cell_type,
    name_to_time,
    dt=0.5,
    min_support=10,
    min_fold=2.0,
    alpha=0.05,
    tol=1e-8,
):
    """build enrichment-tested lineage graph from OT ancestor mappings.

    same output format as build_lineage_graph.
    ancestor_map: cell_name -> {time: ancestor_name}
    name_to_time: cell_name -> float time
    """
    edge_counts = defaultdict(int)
    row_counts = defaultdict(int)
    col_counts = defaultdict(int)
    totals = defaultdict(int)
    observed_counts = defaultdict(int)

    def qtime(x):
        return max(7.5, round(round(x / dt) * dt, 6))

    # OT edges can change with ancestor_map; observed marginals should not.
    # Use all provided observed cells with a known time.
    for cell, ct in name_to_cell_type.items():
        tval = name_to_time.get(cell)
        if tval is None:
            continue
        observed_counts[(ct, qtime(tval))] += 1

    for cell, ancestors in ancestor_map.items():
        if cell not in name_to_cell_type or cell not in name_to_time:
            continue
        ct_cell = name_to_cell_type[cell]
        qt_cell = qtime(name_to_time[cell])

        for t_anc, anc_name in ancestors.items():
            if anc_name not in name_to_cell_type:
                continue
            qt_anc = qtime(t_anc)
            # only count adjacent time steps
            if abs(qt_cell - qt_anc - dt) < tol:
                ct_anc = name_to_cell_type[anc_name]
                s = (ct_anc, qt_anc)
                t = (ct_cell, qt_cell)
                edge_counts[(s, t)] += 1
                row_counts[s] += 1
                col_counts[t] += 1
                totals[qt_anc] += 1

    out = _test_and_filter(edge_counts, row_counts, col_counts, totals,
                           min_support, min_fold, alpha)
    for node, count in observed_counts.items():
        out.add_node(node, observed_count=int(count))
    return out


# ---- plotting helpers ----

def _spectral_seed(G, types):
    """fiedler-vector ordering of the undirected cell-type adjacency graph"""
    n = len(types)
    if n <= 2:
        return list(types)

    idx = {ct: i for i, ct in enumerate(types)}
    W = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        ct_u, ct_v = u[0], v[0]
        if ct_u in idx and ct_v in idx and ct_u != ct_v:
            i, j = idx[ct_u], idx[ct_v]
            W[i, j] += d.get("count", 1)
            W[j, i] += d.get("count", 1)

    L = np.diag(W.sum(axis=1)) - W
    _, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1 if n > 2 else 0]
    return [types[i] for i in np.argsort(fiedler)]


def _minimize_edge_distance(G, types, time_weight_exp=2.0):
    """order cell-type rows to minimise weighted edge distance.

    later timepoint transitions are weighted more heavily via time_weight_exp.
    uses spectral seed -> barycenter refinement -> greedy adjacent swaps.
    """
    n = len(types)
    if n <= 2:
        return list(types)

    type_set = set(types)
    times = sorted({t for _, t in G.nodes()})
    time_idx = {t: i for i, t in enumerate(times)}
    max_ti = max(time_idx.values()) if time_idx else 0

    # collect weighted type-type edges with time emphasis
    edges = []
    for u, v, d in G.edges(data=True):
        ct_u, t_u = u
        ct_v, t_v = v
        if ct_u in type_set and ct_v in type_set:
            ti = max(time_idx.get(t_u, 0), time_idx.get(t_v, 0))
            time_w = ((ti + 1) / (max_ti + 1)) ** time_weight_exp if max_ti > 0 else 1.0
            w = d.get("count", 1) * time_w
            edges.append((ct_u, ct_v, w))

    if not edges:
        return list(types)

    def _total_distance(perm):
        pos = {ct: i for i, ct in enumerate(perm)}
        return sum(w * abs(pos[a] - pos[b]) for a, b, w in edges)

    # spectral seed
    order = _spectral_seed(G, types)
    best_d = _total_distance(order)
    best = list(order)

    # barycenter refinement
    pos = {ct: float(i) for i, ct in enumerate(order)}
    for _ in range(200):
        bary_sum = defaultdict(float)
        bary_wt = defaultdict(float)
        for ct_u, ct_v, w in edges:
            bary_sum[ct_u] += pos[ct_v] * w
            bary_wt[ct_u] += w
            bary_sum[ct_v] += pos[ct_u] * w
            bary_wt[ct_v] += w

        new_order = sorted(
            types,
            key=lambda ct: bary_sum[ct] / bary_wt[ct]
                           if bary_wt[ct] > 0 else pos[ct],
        )
        new_pos = {ct: float(i) for i, ct in enumerate(new_order)}
        if new_pos == pos:
            break
        pos = new_pos

        d = _total_distance(new_order)
        if d < best_d:
            best_d = d
            best = list(new_order)

    # greedy adjacent swaps
    order = list(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            order[i], order[i + 1] = order[i + 1], order[i]
            d = _total_distance(order)
            if d < best_d:
                best_d = d
                best = list(order)
                improved = True
            else:
                order[i], order[i + 1] = order[i + 1], order[i]

    return best


def _filter_by_count(G, min_total_count):
    """keep only cell types whose total observed count meets the threshold"""
    if min_total_count <= 0:
        return G
    totals = defaultdict(int)
    for n, d in G.nodes(data=True):
        totals[n[0]] += d.get("observed_count", 0)

    # fall back to edge-derived totals if observed marginals are not present
    # (or are all zero, as in graphs without observed_count metadata).
    if not totals or all(cnt == 0 for cnt in totals.values()):
        for u, v, d in G.edges(data=True):
            c = d.get("count", 1)
            totals[u[0]] += c
    keep = {ct for ct, cnt in totals.items() if cnt >= min_total_count}
    return G.subgraph([n for n in G.nodes() if n[0] in keep]).copy()


def _make_ribbon(p0, p1, p2, p3, width, n=80):
    """closed polygon tracing a tapered ribbon around a cubic bezier"""
    t = np.linspace(0, 1, n)[:, None]
    mt = 1 - t

    curve = mt**3 * p0 + 3 * mt**2 * t * p1 + 3 * mt * t**2 * p2 + t**3 * p3

    tangent = (
        -3 * mt**2 * p0
        + 3 * (1 - 4 * t + 3 * t**2) * p1
        + 3 * (2 * t - 3 * t**2) * p2
        + 3 * t**2 * p3
    )
    length = np.hypot(tangent[:, 0], tangent[:, 1]).clip(1e-9)
    normal = np.column_stack([-tangent[:, 1] / length, tangent[:, 0] / length])

    taper_min = 0.50
    taper = taper_min + (1 - taper_min) * np.abs(2 * t.flatten() - 1)
    hw = (width / 2) * taper

    upper = curve + hw[:, None] * normal
    lower = curve - hw[:, None] * normal
    return np.vstack([upper, lower[::-1]])


def plot_transition_graph(
    G,
    ax=None,
    node_scale=80,
    color_dict=None,
    min_total_count=0,
    ribbon_width=0.14,
    time_weight_exp=2.0,
):
    """alluvial-style transition graph.

    nodes are (cell_type, time) tuples, edges carry a 'count' attribute.
    times are extracted from node tuples. rows = cell types ordered to
    minimise weighted edge distance, columns = time points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 10), facecolor="white")

    G = _filter_by_count(G, min_total_count)
    types = sorted({ct for ct, _ in G.nodes()})
    times = sorted({t for _, t in G.nodes()})

    if not types:
        ax.text(0.5, 0.5, "No cell types meet the threshold",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#888")
        return ax

    types = _minimize_edge_distance(G, types, time_weight_exp=time_weight_exp)
    row_idx = {ct: i for i, ct in enumerate(types)}

    # map times to evenly spaced x positions
    time_to_x = {t: i for i, t in enumerate(times)}

    pos = {}
    for node in G.nodes():
        ct, t = node
        pos[node] = np.array([float(time_to_x[t]), float(-row_idx[ct])])

    colors = {ct: PALETTE[i % len(PALETTE)] for i, ct in enumerate(types)}
    if color_dict:
        colors.update(color_dict)

    max_w = max((d["count"] for *_, d in G.edges(data=True)), default=1)

    for u, v, d in G.edges(data=True):
        w = d["count"] / max_w
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        dx = x1 - x0

        p0 = np.array([x0, y0])
        p1 = np.array([x0 + 0.42 * dx, y0])
        p2 = np.array([x0 + 0.58 * dx, y1])
        p3 = np.array([x1, y1])

        rw = ribbon_width * (0.20 + 0.80 * w)
        a = 0.35 + 0.55 * w

        verts = _make_ribbon(p0, p1, p2, p3, width=rw)
        fc = to_rgba(colors[u[0]], alpha=a)
        ax.add_patch(Polygon(verts, closed=True, fc=fc, ec="none", zorder=1))

    # node sizes from observed marginals (independent of inferred edges)
    node_counts = {n: int(d.get("observed_count", 0)) for n, d in G.nodes(data=True)}

    # fallback for graphs without observed_count metadata
    if all(c == 0 for c in node_counts.values()):
        node_counts = defaultdict(int)
        for u, v, d in G.edges(data=True):
            c = d.get("count", 1)
            node_counts[u] += c
            node_counts[v] += c

    nodes = list(G.nodes())
    counts = np.array([max(node_counts.get(n, 0), 1) for n in nodes], dtype=float)
    sizes = node_scale * counts / max(counts.max(), 1)

    for node, sz in zip(nodes, sizes):
        ct = node[0]
        x, y = pos[node]
        c = colors[ct]
        ax.scatter(x, y, s=sz * 14, color=to_rgba(c, 0.12),
                   zorder=2, edgecolors="none")
        ax.scatter(x, y, s=sz * 7, color=c,
                   zorder=3, edgecolors="white", linewidths=1.0)

    ax.set_xticks(range(len(times)))
    ax.set_xticklabels([f"{t:.1f}" for t in times], fontsize=8, color="#555")
    ax.set_xlabel("Time", fontsize=10, color="#333", labelpad=8)

    ax.set_yticks([-i for i in range(len(types))])
    ax.set_yticklabels(types, fontsize=9, color="#333")
    ax.set_ylabel("Cell type", fontsize=10, color="#333", labelpad=8)

    ax.margins(0.10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False, bottom=False, colors="#333")
    ax.set_facecolor("white")

    plt.tight_layout()
    return ax