"""Diagnostic visualiser for the record samplers.

Loads the cached mouse-embryo ``TreeRecord`` pickles directly (skipping the
slow AnnData read), runs the configured sampler N times, pulls a few subtrees
at the sample time from each chosen record, and writes a single HTML page
that lets you click through the samples and inspect the picked subtrees and
their leaf cell types.

Edit the constants at the top to swap the sampler / its params / N, then
``python preprocessing/record_sampler_visualizer.py``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.tree_record import TreeRecord  # noqa: E402
from record_samplers.record_samplers import (  # noqa: E402
    AnchorTimeSampler,
    PairedOffsetBackfillSampler,
    PairedOffsetSampler,
    RandomTimeBinSampler,
)


# --------------------------------------------------------------------------
#  USER-EDITABLE CONSTANTS
# --------------------------------------------------------------------------

CACHE_DIR = ROOT / "cache"
DATA_DIR = Path("../data/embryos")
ANNOTATION_FILENAME = "final_subtype.csv"
TYPE_INFO_FILENAME = "final_type_info.csv"

# The pickle filename pattern used by MouseEmbryoDataset is
# ``{trees_filename}_{tree_name}.pkl``. We strip this prefix when loading.
TREES_FILENAME = "final_topology.h5td"

# Which sampler to visualise. Swap the class + kwargs to compare strategies.
# SAMPLER = PairedOffsetBackfillSampler(
#     target_gap=0.5, tol=0.1, sample_time_gap=0.6, backfill_gap=0.5, backfill_tol=0.1,
# )
SAMPLER = PairedOffsetSampler(target_gap=0.5, tol=0.1, sample_time_gap=0.5)
# SAMPLER = AnchorTimeSampler(start_anchor=7.5, end_anchor=8.5, tol=0.1)
# SAMPLER = RandomTimeBinSampler()

N_RECORDS = 2  # number of records the trainer would request
N_SAMPLES = 30  # how many sampler draws to visualise
SUBTREES_PER_RECORD = 3  # how many subtrees to pull per chosen record at sample_time
SEED = 0

# Per-subtree display budget. Subtrees with more nodes are clipped via BFS
# so big snapshots stay legible.
MAX_NODES_PER_SUBTREE = 64

OUT_PATH = ROOT / "analysis" / "results" / "record_sampler_vis.html"


# --------------------------------------------------------------------------
#  Loading
# --------------------------------------------------------------------------


def _load_records(cache_dir: Path, prefix: str) -> dict:
    """Load every ``{prefix}_{tree_name}.pkl`` TreeRecord under cache_dir."""
    prefix_full = f"{prefix}_"
    records: dict[str, TreeRecord] = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not (fname.startswith(prefix_full) and fname.endswith(".pkl")):
            continue
        tree_name = fname[len(prefix_full) : -len(".pkl")]
        records[tree_name] = TreeRecord.load(cache_dir / fname)
    if not records:
        sys.exit(
            f"no cached TreeRecords found in {cache_dir} matching "
            f"prefix {prefix_full!r}"
        )
    return records


def _load_annotations(data_dir: Path) -> dict:
    """Build ``cell_id -> {cell_subtype, cell_type, germ_layer, lineage}``.

    Uses ``final_subtype.csv`` (cell_id -> cell_subtype) joined to
    ``final_type_info.csv`` (cell_subtype -> coarser categories).
    """
    subtype_df = pd.read_csv(data_dir / ANNOTATION_FILENAME, index_col=0)
    type_info = pd.read_csv(data_dir / TYPE_INFO_FILENAME, index_col=0)

    sub_to_type = type_info["cell_type"].to_dict()
    sub_to_gl = type_info["germ_layer"].to_dict()
    sub_to_lineage = type_info["lineage"].to_dict()

    out = {}
    for cell_id, row in subtype_df.iterrows():
        sub = row["cell_subtype"]
        out[cell_id] = {
            "cell_subtype": sub,
            "cell_type": sub_to_type.get(sub, "na"),
            "germ_layer": sub_to_gl.get(sub, "na"),
            "lineage": sub_to_lineage.get(sub, "na"),
        }
    return out


# --------------------------------------------------------------------------
#  Subtree packing
# --------------------------------------------------------------------------


def _bfs_clip(graph, root, max_nodes: int):
    """BFS from root, keep at most ``max_nodes`` nodes."""
    out = [root]
    queue = [root]
    seen = {root}
    while queue and len(out) < max_nodes:
        v = queue.pop(0)
        for c in graph.successors(v):
            if c in seen:
                continue
            if len(out) >= max_nodes:
                break
            seen.add(c)
            out.append(c)
            queue.append(c)
    return out


def _pack_subtree(subtree: dict, annotations: dict, max_nodes: int) -> dict:
    """Convert a TreeRecord subtree dict into JSON-serializable arrays.

    Per-leaf cell-type annotations are pulled from ``annotations`` by node
    name (leaf node names are cell ids). Internal nodes get "na".
    """
    graph = subtree["graph"]
    root = subtree["root"]
    full_size = graph.number_of_nodes()
    truncated = full_size > max_nodes
    if truncated:
        kept = _bfs_clip(graph, root, max_nodes)
        sub = graph.subgraph(kept).copy()
    else:
        sub = graph

    nodes = sorted(
        sub.nodes(),
        key=lambda n: (sub.nodes[n].get("time", 0.0), str(n)),
    )
    idx = {n: i for i, n in enumerate(nodes)}
    parent_idx = [-1] * len(nodes)
    for n in nodes:
        for p in sub.predecessors(n):
            parent_idx[idx[n]] = idx[p]
            break

    times = [float(sub.nodes[n].get("time", float("nan"))) for n in nodes]
    is_leaf = [sub.out_degree(n) == 0 for n in nodes]

    def _ann(n, field):
        if sub.out_degree(n) > 0:
            return "na"
        rec = annotations.get(n)
        return rec[field] if rec else "na"

    return {
        "node_names": [str(n) for n in nodes],
        "parent_idx": parent_idx,
        "times": times,
        "is_leaf": is_leaf,
        "cell_subtype": [_ann(n, "cell_subtype") for n in nodes],
        "cell_type": [_ann(n, "cell_type") for n in nodes],
        "germ_layer": [_ann(n, "germ_layer") for n in nodes],
        "lineage": [_ann(n, "lineage") for n in nodes],
        "root_idx": idx[root] if root in idx else 0,
        "root_name": str(root),
        "tree_name": subtree.get("tree_name", ""),
        "root_time": float(subtree.get("root_time", float("nan"))),
        "leaf_time": float(subtree.get("leaf_time", float("nan"))),
        "full_size": int(full_size),
        "displayed_size": int(len(nodes)),
        "truncated": bool(truncated),
    }


# --------------------------------------------------------------------------
#  Sampling driver
# --------------------------------------------------------------------------


def _sampler_summary(sampler) -> dict:
    cls = type(sampler).__name__
    # only keep simple scalar attributes
    params = {
        k: v for k, v in vars(sampler).items()
        if isinstance(v, (int, float, str, bool)) or v is None
    }
    return {"class": cls, "params": params}


def _build_payload() -> dict:
    print(f"loading records from {CACHE_DIR}")
    records = _load_records(CACHE_DIR, TREES_FILENAME)
    print(f"  loaded {len(records)} records "
          f"(leaf_times: {sorted({r.leaf_time for r in records.values()})})")

    print(f"loading annotations from {DATA_DIR}")
    annotations = _load_annotations(DATA_DIR)
    print(f"  {len(annotations)} cells annotated")

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)  # samplers use legacy np.random
    samples = []
    for s_i in range(N_SAMPLES):
        chosen, sample_time = SAMPLER(records, n_records=N_RECORDS)
        record_entries = []
        for r in chosen:
            # The sampler returns TreeRecord objects; recover the record name
            # by matching object identity against the records dict.
            name = next(
                (k for k, v in records.items() if v is r), "unknown",
            )
            subtrees = r.get_subtrees(sample_time, n=SUBTREES_PER_RECORD, rng=rng)
            record_entries.append({
                "name": name,
                "leaf_time": float(r.leaf_time),
                "n_subtrees_at_bin": len(
                    r.subtree_index.get(r._nearest_bin(sample_time), [])
                ),
                "subtrees": [
                    _pack_subtree(st, annotations, MAX_NODES_PER_SUBTREE)
                    for st in subtrees
                ],
            })
        samples.append({
            "sample_time": float(sample_time),
            "records": record_entries,
        })
        if (s_i + 1) % 10 == 0 or s_i == N_SAMPLES - 1:
            print(f"  drew {s_i + 1}/{N_SAMPLES} samples")

    return {
        "sampler": _sampler_summary(SAMPLER),
        "n_records": int(N_RECORDS),
        "subtrees_per_record": int(SUBTREES_PER_RECORD),
        "max_nodes_per_subtree": int(MAX_NODES_PER_SUBTREE),
        "samples": samples,
    }


# --------------------------------------------------------------------------
#  HTML / JS
# --------------------------------------------------------------------------


_HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>record sampler vis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    :root {
      --bg: #f7f5f0;
      --panel: #ffffff;
      --ink: #1f2530;
      --muted: #6b7280;
      --line: #e6e1d6;
      --accent: #2a4d4a;
      --highlight: #d8541a;
    }
    html, body { background: var(--bg); }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      margin: 18px 22px; color: var(--ink);
      font-size: 13px; letter-spacing: 0.01em;
    }
    h2 {
      font-weight: 500; letter-spacing: -0.015em;
      margin: 0 0 4px 0; font-size: 19px; color: var(--accent);
    }
    .meta {
      color: var(--muted); font-size: 12px; margin-bottom: 14px;
      font-family: 'JetBrains Mono', monospace; letter-spacing: 0;
    }
    .meta b { color: var(--ink); font-weight: 500; }
    .controls {
      display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 12px;
      padding: 12px 14px; background: var(--panel);
      border: 1px solid var(--line); border-radius: 10px;
      align-items: center;
    }
    .controls label { font-size: 12px; color: var(--muted);
                      display: flex; align-items: center; gap: 6px; }
    .controls select, .controls button {
      font-family: inherit; font-size: 12px; color: var(--ink);
      padding: 3px 8px; background: var(--bg);
      border: 1px solid var(--line); border-radius: 4px;
      cursor: pointer;
    }
    .controls input[type=range] { width: 220px; }
    .summary {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;
      font-size: 12px; color: var(--ink);
      font-family: 'JetBrains Mono', monospace;
    }
    .summary b { color: var(--accent); font-weight: 500; }
    .summary .hot { color: var(--highlight); font-weight: 500; }
    .summary table { border-collapse: collapse; margin-top: 8px; }
    .summary td, .summary th {
      padding: 2px 12px 2px 0; font-weight: 400;
      font-size: 11px;
    }
    .summary th { color: var(--muted); font-weight: 500; }
    .grid {
      display: grid;
      gap: 10px;
    }
    .tree-panel {
      background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; padding: 4px;
      min-height: 260px;
    }
    .legend-bar {
      font-size: 11px; color: var(--muted); margin-top: 6px;
      line-height: 1.55;
    }
  </style>
</head>
<body>
  <h2>record sampler / <span id="hdr_sampler"></span></h2>
  <div class="meta" id="meta"></div>

  <div class="controls">
    <button id="btn_prev">&laquo; prev</button>
    <label>sample
      <input type="range" id="sel_idx" min="0" value="0" step="1">
      <span id="lbl_idx" style="font-family: 'JetBrains Mono', monospace;">0</span>
    </label>
    <button id="btn_next">next &raquo;</button>
    <label>color by
      <select id="sel_color">
        <option value="cell_type" selected>cell_type</option>
        <option value="cell_subtype">cell_subtype</option>
        <option value="germ_layer">germ_layer</option>
        <option value="lineage">lineage</option>
      </select>
    </label>
    <button id="btn_resample">resample (re-render)</button>
  </div>

  <div class="summary" id="summary"></div>

  <div class="grid" id="grid"></div>

  <script>
    const DATA = __DATA__;
    const $ = id => document.getElementById(id);

    const PALETTE = [
      '#0072b2', '#e69f00', '#009e73', '#cc79a7', '#56b4e9',
      '#d55e00', '#b58900', '#6c71c4', '#2aa198', '#859900',
      '#268bd2', '#d33682', '#cb4b16', '#586e75', '#073642',
      '#93a1a1', '#dc322f', '#719e5f', '#a67f59', '#7a6fa0',
      '#3b8686', '#c9b86a', '#b75d69', '#8a7ca8', '#5b9bd5',
    ];
    const NA_COLOR = '#c8c2b3';
    const INTERNAL_COLOR = '#a09989';
    const NORMAL_EDGE = '#b9b2a0';

    const BASE_FONT = {
      family: "'Inter', sans-serif", size: 10, color: '#1f2530',
    };

    // Build a single global color map per categorical field so the same
    // value gets the same color across every sample's subtree.
    function buildGlobalCmaps() {
      const cmaps = {};
      for (const key of ['cell_type', 'cell_subtype', 'germ_layer', 'lineage']) {
        const seen = new Set();
        for (const s of DATA.samples) {
          for (const rec of s.records) {
            for (const st of rec.subtrees) {
              for (const v of st[key]) {
                if (v != null && v !== 'na' && v !== 'None') seen.add(v);
              }
            }
          }
        }
        const sorted = Array.from(seen).sort();
        const m = {};
        sorted.forEach((v, i) => { m[v] = PALETTE[i % PALETTE.length]; });
        cmaps[key] = m;
      }
      return cmaps;
    }
    const CMAPS = buildGlobalCmaps();
    function colorFor(field, v) {
      if (v == null || v === 'na' || v === 'None') return NA_COLOR;
      return CMAPS[field][v] || NA_COLOR;
    }

    // Tidy-tree-ish layout: x is leaf order from a DFS that sorts children
    // by time; internal nodes sit at the mean x of their children.
    function layoutSubtree(parent_idx, times) {
      const n = parent_idx.length;
      const ch = Array.from({length: n}, () => []);
      let root = -1;
      for (let i = 0; i < n; i++) {
        if (parent_idx[i] < 0) root = i;
        else ch[parent_idx[i]].push(i);
      }
      if (root < 0) {
        let best = 0;
        for (let i = 1; i < n; i++) if (times[i] < times[best]) best = i;
        root = best;
      }
      const x = new Array(n).fill(NaN);
      const stack = [root];
      const visited = new Array(n).fill(false);
      const leafOrder = [];
      while (stack.length) {
        const v = stack[stack.length - 1];
        if (!visited[v]) {
          visited[v] = true;
          if (ch[v].length === 0) { leafOrder.push(v); stack.pop(); }
          else {
            const sorted = ch[v].slice().sort(
              (a, b) => times[a] - times[b]
            );
            for (let k = sorted.length - 1; k >= 0; k--) stack.push(sorted[k]);
          }
        } else { stack.pop(); }
      }
      leafOrder.forEach((li, k) => { x[li] = k; });
      function assign(v) {
        if (ch[v].length === 0) return;
        for (const c of ch[v]) if (isNaN(x[c])) assign(c);
        let s = 0;
        for (const c of ch[v]) s += x[c];
        x[v] = s / ch[v].length;
      }
      assign(root);
      return x;
    }

    function shortName(name, n) {
      if (name.length <= n) return name;
      return name.slice(0, n - 1) + '…';
    }

    function renderTreePanel(divId, subtree, recordEntry, sampleTime, colorField) {
      const parent = subtree.parent_idx;
      const times = subtree.times;
      const isLeaf = subtree.is_leaf;
      const names = subtree.node_names;
      const xs = layoutSubtree(parent, times);
      const N = parent.length;

      // edges
      const ex = [], ey = [];
      for (let i = 0; i < N; i++) {
        if (parent[i] < 0) continue;
        ex.push(xs[parent[i]], xs[i], null);
        ey.push(times[parent[i]], times[i], null);
      }

      // node groups by leaf/internal so we can style them separately
      const lf = {x: [], y: [], text: [], color: []};
      const it = {x: [], y: [], text: [], color: []};
      for (let i = 0; i < N; i++) {
        const annot = subtree[colorField][i];
        const tip =
          '<b>' + shortName(names[i], 40) + '</b><br>' +
          't = ' + times[i].toFixed(3) + '<br>' +
          (isLeaf[i]
            ? 'cell_subtype: ' + subtree.cell_subtype[i] + '<br>' +
              'cell_type: ' + subtree.cell_type[i] + '<br>' +
              'germ_layer: ' + subtree.germ_layer[i] + '<br>' +
              'lineage: ' + subtree.lineage[i]
            : '[internal node]');
        const bucket = isLeaf[i] ? lf : it;
        bucket.x.push(xs[i]);
        bucket.y.push(times[i]);
        bucket.text.push(tip);
        bucket.color.push(isLeaf[i] ? colorFor(colorField, annot) : INTERNAL_COLOR);
      }

      const traces = [
        { x: ex, y: ey, mode: 'lines', type: 'scatter',
          line: {color: NORMAL_EDGE, width: 1}, hoverinfo: 'skip',
          showlegend: false },
      ];
      if (it.x.length) {
        traces.push({
          x: it.x, y: it.y, text: it.text,
          mode: 'markers', type: 'scatter',
          marker: {size: 6, symbol: 'square', color: it.color,
                   line: {color: '#1f2530', width: 0.4}},
          hovertemplate: '%{text}<extra></extra>',
          showlegend: false,
        });
      }
      if (lf.x.length) {
        traces.push({
          x: lf.x, y: lf.y, text: lf.text,
          mode: 'markers', type: 'scatter',
          marker: {size: 9, symbol: 'circle', color: lf.color,
                   line: {color: '#1f2530', width: 0.4}},
          hovertemplate: '%{text}<extra></extra>',
          showlegend: false,
        });
      }

      // sample_time horizontal reference
      const xMin = Math.min(...xs), xMax = Math.max(...xs);
      traces.push({
        x: [xMin - 0.5, xMax + 0.5], y: [sampleTime, sampleTime],
        mode: 'lines', type: 'scatter',
        line: {color: '#d8541a', width: 1, dash: 'dot'},
        hoverinfo: 'skip', showlegend: false,
      });

      // record's leaf_time horizontal reference
      traces.push({
        x: [xMin - 0.5, xMax + 0.5],
        y: [recordEntry.leaf_time, recordEntry.leaf_time],
        mode: 'lines', type: 'scatter',
        line: {color: '#2a4d4a', width: 1, dash: 'dot'},
        hoverinfo: 'skip', showlegend: false,
      });

      const titleText =
        recordEntry.name +
        '   /   root t=' + subtree.root_time.toFixed(2) +
        '   /   leaves=' + (subtree.is_leaf.filter(b => b).length) +
        (subtree.truncated
          ? '   /   <span style="color:#d8541a">(clipped ' +
            subtree.displayed_size + '/' + subtree.full_size + ')</span>'
          : '');

      const layout = {
        title: {text: titleText, font: {size: 11, color: '#1f2530'}},
        font: BASE_FONT,
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        xaxis: {showticklabels: false, zeroline: false,
                gridcolor: 'rgba(0,0,0,0)'},
        yaxis: {title: {text: 'time', font: {size: 10, color: '#6b7280'}},
                autorange: 'reversed', gridcolor: '#eee9dc'},
        margin: {l: 42, b: 18, r: 8, t: 30},
        showlegend: false,
      };
      Plotly.react(divId, traces, layout, {displayModeBar: false});
    }

    function renderLegend(divId, colorField, valuesPresent) {
      const vals = Array.from(valuesPresent).filter(
        v => v != null && v !== 'na' && v !== 'None'
      ).sort();
      const html = vals.map(v =>
        '<span style="display:inline-flex; align-items:center; gap:4px; ' +
        'margin-right:10px;">' +
        '<span style="display:inline-block; width:10px; height:10px; ' +
        'background:' + colorFor(colorField, v) + '; border-radius:2px; ' +
        'border:1px solid #1f2530;"></span>' + v + '</span>'
      ).join('');
      $(divId).innerHTML =
        '<b>' + colorField + ':</b> ' + (html || '(no leaves)');
    }

    function summarizeLeaves(record, field) {
      const counts = {};
      let total = 0;
      for (const st of record.subtrees) {
        for (let i = 0; i < st.is_leaf.length; i++) {
          if (!st.is_leaf[i]) continue;
          const v = st[field][i] ?? 'na';
          counts[v] = (counts[v] || 0) + 1;
          total += 1;
        }
      }
      const sorted = Object.entries(counts).sort((a, b) => b[1] - a[1]);
      return {counts: sorted, total};
    }

    function renderSample(idx) {
      const s = DATA.samples[idx];
      const colorField = $('sel_color').value;
      $('lbl_idx').textContent = (idx + 1) + ' / ' + DATA.samples.length;

      // summary block
      let html =
        '<b>sample #' + idx + '</b>  &nbsp;  ' +
        '<b>sample_time</b> <span class="hot">' +
        s.sample_time.toFixed(3) + '</span>  &nbsp;  ' +
        '<b>n_records</b> ' + s.records.length;

      html += '<table>';
      html += '<tr><th>record</th><th>leaf_time</th><th>gap to sample_time</th>' +
              '<th>pool @ sample bin</th><th>shown</th>' +
              '<th>top ' + colorField + 's (leaves across shown subtrees)</th></tr>';
      for (const rec of s.records) {
        const summary = summarizeLeaves(rec, colorField);
        const topTxt = summary.counts.slice(0, 5).map(
          ([v, c]) => v + ' ' + c
        ).join(', ') + (summary.counts.length > 5 ? ', …' : '');
        const gap = (rec.leaf_time - s.sample_time);
        html += '<tr>' +
          '<td><b>' + rec.name + '</b></td>' +
          '<td>' + rec.leaf_time.toFixed(3) + '</td>' +
          '<td>+' + gap.toFixed(3) + '</td>' +
          '<td>' + rec.n_subtrees_at_bin + '</td>' +
          '<td>' + rec.subtrees.length + '</td>' +
          '<td>' + topTxt + '</td>' +
          '</tr>';
      }
      html += '</table>';
      $('summary').innerHTML = html;

      // grid: rows = records, cols = subtrees per record
      const grid = $('grid');
      const R = s.records.length;
      const K = Math.max(1, ...s.records.map(r => r.subtrees.length));
      grid.style.gridTemplateColumns = 'repeat(' + K + ', minmax(0, 1fr))';
      grid.innerHTML = '';

      const present = new Set();
      const panels = [];
      for (let r = 0; r < R; r++) {
        const rec = s.records[r];
        for (let k = 0; k < K; k++) {
          const cell = document.createElement('div');
          cell.className = 'tree-panel';
          const divId = 'tree_' + r + '_' + k;
          cell.id = divId;
          grid.appendChild(cell);
          if (k < rec.subtrees.length) {
            const st = rec.subtrees[k];
            panels.push({divId, st, rec});
            for (let i = 0; i < st.is_leaf.length; i++) {
              if (st.is_leaf[i]) present.add(st[colorField][i]);
            }
          }
        }
      }
      // legend bar at the bottom
      const legend = document.createElement('div');
      legend.id = 'legend_bar';
      legend.className = 'legend-bar';
      legend.style.gridColumn = '1 / -1';
      grid.appendChild(legend);

      for (const p of panels) {
        renderTreePanel(p.divId, p.st, p.rec, s.sample_time, colorField);
      }
      renderLegend('legend_bar', colorField, present);
    }

    function init() {
      const samp = DATA.sampler;
      const paramStr = Object.entries(samp.params)
        .map(([k, v]) => k + '=' + v).join('  ');
      $('hdr_sampler').textContent = samp.class;
      $('meta').innerHTML =
        '<b>params</b> ' + (paramStr || '(none)') +
        '  &nbsp; <b>n_records</b> ' + DATA.n_records +
        '  &nbsp; <b>subtrees/record</b> ' + DATA.subtrees_per_record +
        '  &nbsp; <b>n_samples</b> ' + DATA.samples.length +
        '  &nbsp; <b>max_nodes/subtree</b> ' + DATA.max_nodes_per_subtree +
        '  &nbsp; dotted lines: <span style="color:#d8541a">sample_time</span> /' +
        ' <span style="color:#2a4d4a">record.leaf_time</span>';

      const slider = $('sel_idx');
      slider.max = DATA.samples.length - 1;
      slider.addEventListener('input', () => renderSample(parseInt(slider.value)));
      $('btn_prev').addEventListener('click', () => {
        slider.value = Math.max(0, parseInt(slider.value) - 1);
        renderSample(parseInt(slider.value));
      });
      $('btn_next').addEventListener('click', () => {
        slider.value = Math.min(
          DATA.samples.length - 1, parseInt(slider.value) + 1,
        );
        renderSample(parseInt(slider.value));
      });
      $('sel_color').addEventListener('change',
        () => renderSample(parseInt(slider.value)));
      $('btn_resample').addEventListener('click',
        () => renderSample(parseInt(slider.value)));
      renderSample(0);
    }

    init();
  </script>
</body>
</html>
"""


# --------------------------------------------------------------------------
#  Main
# --------------------------------------------------------------------------


def main():
    payload = _build_payload()
    js_data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    html = _HTML_TEMPLATE.replace("__DATA__", js_data)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(html)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"wrote {OUT_PATH}  ({size_mb:.2f} MB)")
    print(f"open it in a browser: file://{OUT_PATH}")


if __name__ == "__main__":
    main()
