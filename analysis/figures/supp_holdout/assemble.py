"""render the metrics panel and nest it into imgs/supp_holdout.{svg,pdf,png}. see concept.md.

    python -m analysis.figures.supp_holdout.assemble
"""
import os

from analysis.figures.supp_holdout import plot_metrics
from analysis.figures.supp_holdout import style as S
from utils import svg_assembly as sa


def main():
    PT = S.PT_PER_IN
    path, (w, h) = plot_metrics.render()
    assert w <= S.PAGE_W_IN, f"{w:.2f} in is wider than the {S.PAGE_W_IN} in page"
    master = sa.new_master()
    master.append(sa.nest_panel(path, w, h, 0, 0, 0))
    sa.write_master(master, os.path.join(S.IMGS, "supp_holdout.svg"), w * PT, h * PT)


if __name__ == "__main__":
    main()
