"""train one genot variant on the train split.

runs the paper's script (baselines/genot_*.py) with three lines swapped: the dataset
(held-out clones removed), OUTPUT_PATH (into RESULTS/genot/) and the variant's
constant. each swap must match exactly once, so the rest is the paper's code.

    python -m analysis.figures.supp_holdout.train_genot genot_cbfgw_fp0.1
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from analysis.figures.supp_holdout import common as C

name = sys.argv[1]
script, line, new_line = C.GENOT_VARIANTS[name]
out = os.path.join(C.RESULTS, "genot", f"{name}.pkl")
os.makedirs(os.path.dirname(out), exist_ok=True)

with open(script) as f:
    src = f.read()
out_line = next(l for l in src.splitlines() if l.startswith("OUTPUT_PATH = "))
for a, b in [(line, new_line),
             ("dataset = MouseEmbryoDataset()", f"dataset = HeldOutMouseEmbryoDataset({C.HOLDOUT!r})"),
             (out_line, f"OUTPUT_PATH = {out!r}")]:
    assert src.count(a) == 1, (script, a)
    src = src.replace(a, b)
assert "outputs/e_baselines" not in src

print(f"{name}: {script} with '{new_line}' -> {out}", flush=True)
path = os.path.join(ROOT, script)
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path,
                                  "HeldOutMouseEmbryoDataset": C.HeldOutMouseEmbryoDataset})
