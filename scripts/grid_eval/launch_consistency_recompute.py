#!/usr/bin/env python
"""bring every grid run's eval artefacts in line with the main-text figure's eval.

derives the work from what is missing or stale under outputs/ for the grid
(analysis.figures.grid_summary.compute_grid.grid_configs) and submits it in
dependency order:

  1. couplings  -- the sampling side at its canonical n (8 for the bridges, the
                   main-text figure's setting; 1 for the one-step splitters) and the
                   pool side at n=1, wherever the pkl is missing (one job each)
  2. distortion -- the distortion CSVs via compute_distortion (clonal + marginal +
                   descendant with the splitter, marginal_pool with the imputing
                   model) for every (run, step) lacking fresh CSVs: only the missing
                   ones unless --force-distortion (batched; the descendant pass
                   starts from the sampling-side coupling, so after 1)
  3. refresh    -- every run's metrics CSVs re-scored with the figure's code
                   (batched; after 1)
  4. maps       -- the sampling-side transition graphs of the similarity heatmap's
                   runs, at the figure's PMI quantile, where missing (--force-maps
                   rebuilds all; batched; after 1 and 3, since the refresh and the
                   maps write the same metrics CSVs)
  --then-render   a final job that rebuilds the caches + figures after everything

    python scripts/grid_eval/launch_consistency_recompute.py --dry-run
    python scripts/grid_eval/launch_consistency_recompute.py --smoke     # one job per kind
    python scripts/grid_eval/launch_consistency_recompute.py
"""
from _common import ROOT
import argparse
import json
import os
import subprocess
import time

from analysis.figures.grid_summary import style as S
from analysis.figures.grid_summary.compute_grid import grid_configs
from analysis.lib.eval_core.runner import DISTORTIONS, assignments_path, distortion_path

HERE = os.path.dirname(os.path.abspath(__file__))


def _fresh_distortion(d, cfg, step):
    """the four CSVs exist, the clonal one in the figure's definition and the
    marginal one with its cell-type TV column."""
    paths = [distortion_path(d, cfg, step, w) for w in DISTORTIONS]
    if not all(os.path.exists(p) for p in paths):
        return False
    with open(paths[0]) as f:
        fresh = "method" in f.readline()
    with open(paths[1]) as f:
        return fresh and "cell_type_tv" in f.readline()


EXTRA_DEPS = []
SBATCH_ARGS = []      # e.g. a partition / account override, prepended to every sbatch


def submit(script, name, env, deps=(), dry=False):
    cmd = ["sbatch", "--parsable", *SBATCH_ARGS, f"--job-name={name}"]
    deps = [*deps, *EXTRA_DEPS]
    if deps:
        cmd.append("--dependency=afterany:" + ":".join(deps))
    cmd.append(os.path.join(HERE, script))
    if dry:
        print(f"  would submit {name}  {script}  {dict(env)}"[:220])
        return f"dry{abs(hash(name)) % 100000}"
    jid = subprocess.run(cmd, check=True, capture_output=True, text=True,
                         env={**os.environ, **env}).stdout.strip().split(";")[0]
    print(f"  SUBMITTED {name}  jid={jid}")
    return jid


def chunks(xs, k):
    return [xs[i:i + k] for i in range(0, len(xs), k)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, nargs="+", default=list(S.STEPS))
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="one job per kind, one item each")
    ap.add_argument("--only", action="append",
                    choices=["coupling", "distortion", "refresh", "maps"],
                    help="repeatable; submit only these kinds (dependencies on kinds not "
                         "submitted are dropped)")
    ap.add_argument("--after", nargs="*", default=[],
                    help="extra slurm job ids every submitted job waits for (afterany)")
    ap.add_argument("--then-render", action="store_true",
                    help="submit render_grid.sbatch after every job submitted here")
    ap.add_argument("--sbatch", default="",
                    help="extra sbatch options for every job, e.g. "
                         "'--account=mzitnik_lab --partition=gpu_requeue --constraint=h200|h100|a100' "
                         "to run outside the Kempner GPU cap (the batch jobs are idempotent, "
                         "so a preemptible partition is safe)")
    ap.add_argument("--force-distortion", action="store_true",
                    help="recompute clonal + marginal distortion for every run (e.g. after a "
                         "change of summary_qc.style.SPLIT_STEPS); by default only missing / "
                         "stale CSVs are computed")
    ap.add_argument("--force-maps", action="store_true",
                    help="rebuild every transition map (e.g. after a PMI cutoff change); "
                         "by default only missing maps are built")
    args = ap.parse_args()
    dry = args.dry_run
    B = 1 if args.smoke else args.batch_size
    kinds = set(args.only or ["coupling", "distortion", "refresh", "maps"])
    EXTRA_DEPS.extend(args.after)
    SBATCH_ARGS.extend(args.sbatch.split())

    runs = []   # (dir, cfg, bridge, pool, aligner, step)
    for step in args.steps:
        for r in grid_configs(step).itertuples(index=False):
            runs.append((os.path.join(S.OUTPUTS, r.cfg), r.cfg, r.bridge, r.pool, r.aligner, step))
    print(f"{len(runs)} (run, step) tuples over steps {args.steps}")

    # 1. couplings
    coupling = []
    for d, cfg, bridge, *_, step in runs:
        n = S.coupling_n(bridge)
        if not os.path.exists(assignments_path(d, cfg, step, n, True)):
            coupling.append(("eval_coupling_splitter.sbatch", f"cplspl_{cfg[34:74]}_{step}_n{n}",
                             {"EXPERIMENT_DIR": d, "STEP": str(step), "POOL_N_STEPS": str(n)}))
        if not os.path.exists(assignments_path(d, cfg, step, 1, False)):
            coupling.append(("eval_coupling.sbatch", f"cpl_{cfg[34:74]}_{step}_n1",
                             {"EXPERIMENT_DIR": d, "STEP": str(step)}))
    # 2. distortion, with the sampling-side coupling's n for the descendant pass
    dist = [f"{d}|{step}|{S.coupling_n(bridge)}" for d, cfg, bridge, *_, step in runs
            if args.force_distortion or not _fresh_distortion(d, cfg, step)]
    # 3. refresh: every run
    refresh = sorted({d for d, *_ in runs})
    # 4. maps: the heatmap's runs, sampling side at the canonical n
    maps = []
    for d, cfg, bridge, pool, aligner, step in runs:
        if not S.in_heatmap(bridge, pool, grid_configs(step).set_index("cfg").at[cfg, "pairing"], aligner):
            continue
        n = S.coupling_n(bridge)
        built = os.path.join(d, f"cell_type_map_{cfg}_step{step}_adj_n{n}_splitter.pkl")
        if args.force_maps or not os.path.exists(built):
            maps.append(assignments_path(d, cfg, step, n, True))
    if args.smoke:
        coupling, dist, refresh, maps = coupling[:1], dist[:1], refresh[:1], maps[:1]
    if "coupling" not in kinds:
        coupling = []
    if "distortion" not in kinds:
        dist = []
    if "refresh" not in kinds:
        refresh = []
    if "maps" not in kinds:
        maps = []
    print(f"work: {len(coupling)} couplings, {len(dist)} distortion tuples, "
          f"{len(refresh)} runs to refresh, {len(maps)} maps")

    jids = {"coupling": [], "distortion": [], "refresh": [], "maps": [], "render": []}
    print("\n1. couplings")
    for script, name, env in coupling:
        jids["coupling"].append(submit(script, name, env, dry=dry))
    print("\n2. distortion (after the couplings)")
    for i, batch in enumerate(chunks(dist, B)):
        jids["distortion"].append(submit(
            "eval_distortion_batch.sbatch", f"dist_batch_{i}",
            {"BATCH": ";".join(batch), "OVERWRITE": "1" if args.force_distortion else "0"},
            deps=jids["coupling"], dry=dry))
    print("\n3. refresh (after the couplings)")
    for i, batch in enumerate(chunks(refresh, max(1, B // 2))):
        jids["refresh"].append(submit(
            "refresh_metrics_batch.sbatch", f"refresh_batch_{i}",
            {"BATCH": ";".join(batch), "STEPS": " ".join(map(str, args.steps))},
            deps=jids["coupling"], dry=dry))
    print("\n4. transition maps (after the couplings and the refresh)")
    for i, batch in enumerate(chunks(maps, B)):
        jids["maps"].append(submit("eval_cell_type_map_batch.sbatch", f"ctmap_batch_{i}",
                                   {"BATCH": ";".join(batch)},
                                   deps=jids["coupling"] + jids["refresh"], dry=dry))
    if args.then_render:
        print("\n5. render (after everything)")
        everything = [j for k, v in jids.items() if k != "render" for j in v]
        jids["render"].append(submit("render_grid.sbatch", "render_grid",
                                     {"STEPS": " ".join(map(str, args.steps))},
                                     deps=everything, dry=dry))

    if not dry:
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        out = os.path.join(ROOT, "logs", f"consistency_recompute_{time.strftime('%Y%m%d_%H%M%S')}.json")
        with open(out, "w") as f:
            json.dump({"steps": args.steps, "jobs": jids, "work": {
                "coupling": [n for _, n, _ in coupling], "distortion": dist,
                "refresh": refresh, "maps": maps}}, f, indent=1)
        print(f"\nmanifest -> {out}")


if __name__ == "__main__":
    main()
