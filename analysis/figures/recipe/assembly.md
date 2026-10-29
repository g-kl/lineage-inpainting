fate recipe figure assembly notes

this folder builds the panel for the fate-specifying-tf figure. the panel is the
radial subway map.

pipeline, upstream to downstream

1 analysis/pipelines/recipe/fate_tfs.py reads the tf_projection_screen_ko
  tree pickles and writes a csv with one row per (group, time bin, tf) carrying
  penetrance, specificity, n_disrupted, n_leaves and a per (group, time)
  variability score. the group is whatever --level says: --level cell_type (the
  default) writes fate_tfs.csv, --level lineage writes fate_tfs_lineage.csv. the
  same machinery just buckets leaves by a different annotation column, so the
  lineage numbers are computed from the trees directly, not summed from the
  cell-type csv (that csv is prefiltered, so summing it would be wrong). this is
  the heavy compute over the trees; rerun only when the screen changes. both csvs
  are needed by the panel, so run it twice: once plain, once with --level lineage.

2 subway_data.py is the panel-specific compute step. it reads fate_tfs.csv,
  fate_tfs_lineage.csv and the cell-type annotation (final_subtype.csv and
  final_type_info.csv, only the cell_subtype column is loaded so the big matrix
  is never touched). it picks the 48 (N_TYPES) most abundant cell types and
  orders them into lineage groups sorted by germ layer. for the three outer time
  bins (e8.0, e8.5, e9.0) it keeps, per cell type, the top 3 tfs passing the
  penetrance cutoff and the outer specificity cutoff. for the inner bin (e7.5) it
  instead works at lineage level and keeps the top 3 per lineage by specificity,
  with its own (lower) inner specificity cutoff. it does the outer selection
  twice, once ranking by penetrance and once by specificity, and writes
  subway_data_penetrance.json and subway_data_specificity.json. all cutoffs and
  counts are constants at the top of the file (the inner and outer specificity
  cutoffs are separate because the inner lineage level has a weaker specificity
  signal).

  why the inner bin is lineage level and specificity only: at e7.5 the
  divergences are broad, so at cell-type level there is little signal, and at
  lineage level penetrance is dominated by genes whose knockout perturbs
  essentially everything early (hmga2, p4hb, ssbp3 ...) with penetrance near 1
  for every lineage. only specificity surfaces the lineage-specifying tfs (e.g.
  endoderm foxa1/foxa2, blood-endothelial mesoderm tal1/sox7). big lineages that
  make up much of the whole-embryo background (neural ectoderm) have no early
  specifier and so get an empty block. INNER_RANK_BY in subway_data.py toggles
  this.

3 subway_plot.py is pure drawing. it reads each subway_data_<rank>.json and
  renders the radial map, writing subway_<rank>.pdf/.svg/.png for both rankings.
  no data work happens here so cosmetics can be changed and re-rendered fast. all
  style and geometry knobs (including the per-lineage colour palette) are
  constants at the top of the file.

what the panel shows

the inner band is a ring of per-lineage arc blocks holding each lineage's
e7.5->e8.0 specifying tfs. from the e8.0 ring outward each cell type becomes its
own spoke to the rim (e9.0), carrying its later tfs. four light dotted concentric
rings mark the 0.5-spaced time bins. each tf is its gene name on a small white box
written perpendicular across the spoke at the radius of its divergence time.
spokes and blocks are coloured by lineage (one distinct hue each); the cell type
name sits just outside the rim in its lineage colour. the lineage legend lives in
a strip to the left of the circle so it never overlaps the map. there are two
versions, identical inner band, differing only in whether the outer top-3 tfs were
chosen by penetrance or specificity.

regenerate

run, with the lt conda env:
  python analysis/pipelines/recipe/fate_tfs.py
  python analysis/pipelines/recipe/fate_tfs.py --level lineage
  python analysis/figures/recipe/subway_data.py
  python analysis/figures/recipe/subway_plot.py
the fate_tfs.py steps only need rerunning when the screen or its cutoffs change.
for cosmetic tweaks just rerun subway_plot.py.

tuning notes

the knobs that trade station density for legibility are LABEL_FS, GROUP_GAP,
HUB_R and the per-bin TOP_PER_BIN cutoff in subway_data.py. labels are written
perpendicular to the spoke so their radial footprint is just the font height and
consecutive stations on a spoke do not collide; the tangential footprint is the
gene name, which can crowd neighbouring spokes. alternating spokes have their
label radii nudged in and out by STAGGER (subway_plot.py) so adjacent spokes'
labels interleave radially. radius encodes developmental time, not distance, so
the hub is just a visual start point. rim labels read outward on both halves of
the circle (rotation flips past vertical). the lineage colour palette is defined
in subway_plot.py. gene names are set in computer modern typewriter via mathtext
($\mathtt{...}$) at TF_FS, a little smaller than the other labels; hyphens are
swapped for a unicode hyphen so names like nkx2-5 are not typeset as a spaced
math minus (there is no system latex, so real \texttt / usetex is not used).
