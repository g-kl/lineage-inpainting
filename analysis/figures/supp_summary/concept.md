# concept (cell type summary supplement)

how lineage inpainting's cell type transition map agrees with the qiu tree (`literature/Qiu24_mouse_tree_edges.csv`). the scoring is `compute_verdicts.py` and the vocabulary map is `celltype_map.csv`. blood and the extraembryonic lineages are left out on both sides (`style.EXCLUDE`).

- **lineage inpainting vs. qiu**: composition bar chart of all cell type transitions, split into match (true matches, identity transitions, and collapsed within 2 hops), boundary (spatial-continuity edges), reverse (reversed edges) and other mismatch.

- **lineage inpainting vs. qiu at single cell level**: same composition bar chart, but of single cell transitions, in percent.

- **reverse table**: table of top 10 reversed transitions

- **boundary table**: table of top 10 boundary transitions

- **mismatch table**: table of top 10 other mismatch transitions

- **unmapped table**: table of top 10 unmapped transitions

# implementation

compute writes small caches into `numerical_results/`, the plot scripts and `assemble.py` read only those and write into `imgs/` (both gitignored). one gpu for the map, the rest cpu, about a minute.

a lineage inpainting vs. qiu, b the same at single cell level, c reverse table, d boundary table, e mismatch table, f unmapped table.

the map is fig 3's: lineage inpainting's (`summary_qc.style.MODEL_ASSIGNMENTS`) built with the same recipe (`PMI_QUANTILE`, `MIN_COMPOSITION`, `COMPOSITION`, `RESOLUTION_BY_TIME`). every pair within the nn snap cutoff, no pmi cut, no subsample, an edge is kept when its in + out composition is >= 0.2, and E7.5 is labelled by germ layer. the build is deterministic, so this graph and fig 3's are the same.

`style.py`: paths, `EXCLUDE`, `MAX_HOPS`, the category palette, panel sizes, `N_TOP`, `FS_ROW`. the model assignments, `N_TREES`, the map recipe, the rc and the page width come from `analysis/figures/summary_qc/style.py`. the table rows are 8 pt, the one place below 10 pt.

## verdicts

`compute_verdicts.judge`, in order: `identity` (source = target), `unmapped` (an end has no qiu state in `celltype_map.csv`), `collapsed` (both ends share a qiu state), `match` (a qiu path source -> target within `MAX_HOPS`), `reversed` (a qiu path target -> source), else `mismatch`.

## categories

one per type transition, from the verdict, `MAX_HOPS` = 2:

| category | from |
| --- | --- |
| match | `match` or `collapsed`, and every identity type transition |
| reverse | `reversed` |
| boundary | `mismatch` between qiu neighbours: spatial continuity in the unpruned file, or qiu siblings |
| other mismatch | any other `mismatch` |
| unmapped | `unmapped` |

a counts type transitions, b counts cells. the mismatch table is other mismatch only.

## cell type map

`celltype_map.csv` maps each of our cell types to the qiu states it stands for, `;` separated, many-to-many both ways. it is curated, so it is committed. the rows are every label a node can carry: the label, at its time's resolution (`summary_qc.style.RESOLUTION_BY_TIME`), of any cell within one time bin of the node, since the assignments snap within +-1 bin. `propose_map.py` fills `qiu_types` where `norm` (lowercase, no parenthetical, no trailing "cell(s)", singular) hits a qiu state, and leaves the rest blank with difflib `candidates`.

| status | |
| --- | --- |
| `auto` | the `norm` match, accepted as is |
| `curated` | set by hand |
| `none` | reviewed, qiu holds no equivalent state, so it scores `unmapped` |
| `todo` | no `norm` match, not yet reviewed |

a rerun keeps `qiu_types` and `status` and refreshes `candidates`.

## compute -> numerical_results/

`compute_map.py` -> `cell_type_map.pkl` (graph, name_to_labels). `build_cell_type_map` on the model assignments with fig 3's recipe. one gpu, ~7 min, most of it the dataset load.

`compute_verdicts.py` -> `transitions.csv` (source/target type, count, verdict, category, qiu_path), one row per type transition over the whole map, identity included, after the `EXCLUDE` cut. panels a-f. asserts every drawn type has a row in `celltype_map.csv` and every `qiu_types` name is a qiu state.

## plot + assemble -> imgs/

`plot_composition.py` -> a with its legend, b. `plot_lists.py` -> c-f, laid out in inches so each is as wide as its longest label. `assemble.py` -> `imgs/supp_summary.{svg,pdf,png}` through `utils.svg_assembly`: a, then b, then c and d, then e and f.

## run order

    PY=python
    $PY -m analysis.figures.supp_summary.compute_map         # gpu
    $PY -m analysis.figures.supp_summary.compute_verdicts
    $PY -m analysis.figures.supp_summary.assemble

`propose_map.py` only needs rerunning if the vocabulary changes (new annotations, the snap window, or the resolution by time). only assemble needs rerunning for cosmetics.
