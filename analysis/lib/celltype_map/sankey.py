from collections import defaultdict
from itertools import permutations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon


PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#86BCB6", "#8CD17D", "#B6992D", "#499894", "#D37295",
    "#A0CBE8", "#FFBE7D", "#D4A6C8", "#79706E", "#FABFD2",
]


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


def _minimize_edge_distance(G, types, time_weight_exp=2.0, groups=None):
    """order cell-type rows to minimise weighted edge distance.

    later timepoint transitions are weighted more heavily via time_weight_exp.
    uses spectral seed -> barycenter refinement -> greedy adjacent swaps.

    if ``groups`` (ct -> group key, e.g. germ layer) is given, the order is
    constrained so same-group types stay contiguous: the free solution is
    projected onto contiguous blocks (groups placed by their members' mean
    position), then greedy swaps refine only within each block.
    """
    n = len(types)
    if n <= 2:
        return list(types)

    type_set = set(types)
    times = sorted({t for _, t in G.nodes()})
    time_idx = {t: i for i, t in enumerate(times)}
    max_ti = max(time_idx.values()) if time_idx else 0

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

    order = _spectral_seed(G, types)
    best_d = _total_distance(order)
    best = list(order)

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

    if groups is not None:
        # project the free order onto contiguous germ-layer blocks (keeping
        # each block's within-order), then pick the block order that minimises
        # weighted edge distance so adjacent blocks are the ones that talk most.
        members = defaultdict(list)
        for ct in best:
            members[groups.get(ct)].append(ct)
        blocks = _order_blocks(list(members.values()), _total_distance)
        best = [ct for b in blocks for ct in b]
        best_d = _total_distance(best)

    order = list(best)
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            if groups is not None and groups.get(order[i]) != groups.get(order[i + 1]):
                continue  # keep germ-layer blocks contiguous
            order[i], order[i + 1] = order[i + 1], order[i]
            d = _total_distance(order)
            if d < best_d:
                best_d = d
                best = list(order)
                improved = True
            else:
                order[i], order[i + 1] = order[i + 1], order[i]

    return best


def _order_blocks(blocks, total_distance):
    """order contiguous type-blocks to minimise weighted edge distance.

    within-block order is fixed; only the block sequence varies. brute-forces
    all permutations when few blocks (germ layers), else greedy block swaps.
    """
    if len(blocks) <= 1:
        return list(blocks)
    if len(blocks) <= 7:
        return list(min(permutations(blocks),
                        key=lambda bl: total_distance([c for b in bl for c in b])))

    order = list(blocks)
    best, best_d = list(order), total_distance([c for b in order for c in b])
    improved = True
    while improved:
        improved = False
        for i in range(len(order) - 1):
            order[i], order[i + 1] = order[i + 1], order[i]
            d = total_distance([c for b in order for c in b])
            if d < best_d:
                best_d, best, improved = d, list(order), True
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
    rows = cell types ordered to minimise weighted edge distance,
    columns = time points.
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

    node_counts = {n: int(d.get("observed_count", 0)) for n, d in G.nodes(data=True)}

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
