"""render the grid-summary figures. see concept.md.

each figure is one page-width panel, nested into a master svg and written with its
pdf + png siblings into figures/<step>/<name>.{svg,pdf,png} (committed); the panel
svgs land in imgs/<step>/panels/ (gitignored).

    python -m analysis.figures.grid_summary.assemble                    # every figure, style.STEP
    python -m analysis.figures.grid_summary.assemble --step 5000
    python -m analysis.figures.grid_summary.assemble --figure stopgrad --figure ct_similarity
"""
import argparse
import os

from analysis.figures.grid_summary import (
    plot_ablations, plot_lollipops, plot_sampler_vs_imputer, plot_similarity,
)
from analysis.figures.grid_summary import style as S
from utils import svg_assembly as sa

# figure name -> renderer(panels_dir, step) returning (svg path, (w_in, h_in))
FIGURES = {
    **{name: (lambda d, s, n=name: plot_ablations.render(d, n, s))
       for name in plot_ablations.ABLATIONS},
    "aligner_ablation": lambda d, s: plot_lollipops.render(d, "aligner_ablation", s),
    "ct_similarity": plot_similarity.render,
    "sampler_vs_imputer": plot_sampler_vs_imputer.render,
}


def assemble(name, step):
    panels = os.path.join(S.imgs_dir(step), "panels")
    os.makedirs(panels, exist_ok=True)
    path, (w_in, h_in) = FIGURES[name](panels, step)
    PT = S.PT_PER_IN
    pad = S.PAGE_PAD_IN * PT
    master = sa.new_master()
    master.append(sa.nest_panel(path, w_in, h_in, pad, pad, 0))
    return sa.write_master(master, os.path.join(S.figures_dir(step), f"{name}.svg"),
                           w_in * PT + 2 * pad, h_in * PT + 2 * pad)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=S.STEP)
    ap.add_argument("--figure", action="append", choices=list(FIGURES),
                    help="repeatable; default every figure")
    args = ap.parse_args()
    for name in (args.figure or FIGURES):
        assemble(name, args.step)


if __name__ == "__main__":
    main()
