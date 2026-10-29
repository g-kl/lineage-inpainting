#!/usr/bin/env python
"""build the binned per-cell SNP counts from the allSNP / read.BC pairs.

allSNP keys on read name, read.BC maps read name to barcode, and both are sorted by it, so
the two are merged with `join` and aggregated in one streaming pass per lane. no UMI
column, so counts are per read, taken at the read's first SNP.

counts.npz and histogram_f<FINE>.csv are prebuilt caches, so this only needs running to
rebuild them. reading each lane twice is the slow half.

    python -m analysis.pipelines.tbx5.demultiplexing.aggregate
"""
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

from analysis.pipelines.tbx5.demultiplexing import common as C

# the bin edges are cut on read mass over both Tbx5 lanes.
HIST_LANES = [f"{C.TARGET}_{g}" for g in C.GEMS]

# one count per read into a FINE-wide bin of the cumulative coordinate. barcodes are not
# needed for this, so it reads allSNP alone.
HIST_AWK = r"""
BEGIN { FS = "\t" }
$1 == "." { next }
$2 != p { p = $2; split($1, c, "."); split($3, q, ";")
          h[c[1] "\t" int(q[1] / FINE)]++ }
END { for (k in h) print k "\t" h[k] }
"""

# same one-count-per-read rule with the barcode joined on. emits two ints per read, a cell
# row and a feature column, so nothing has to be held in memory.
COUNT_AWK = r"""
BEGIN { FS = "\t"
  while ((getline l < LUT) > 0) { split(l, z, "\t"); lut[z[1]] = z[2] + 0 }
  while ((getline l < BCS) > 0) { bidx[l] = n++ }
}
$1 != p { p = $1
  if ($4 in bidx) {
    split($3, q, ";"); f = int(q[1] / FINE)
    if (f in lut) print bidx[$4] "\t" (2 * lut[f] + ($2 ~ /G2$/ ? 1 : 0))
  }
}
"""


def path(lane, kind):
    return os.path.join(C.SNP_DIR, f"{lane}_{kind}.tsv.gz")


def histogram():
    """reads per (chromosome, FINE bin) over both lanes, cached."""
    out = C.out(f"histogram_f{C.FINE}.csv")
    if os.path.exists(out):
        return pd.read_csv(out)
    os.makedirs(C.OUT, exist_ok=True)
    parts = []
    for lane in HIST_LANES:
        z = subprocess.Popen(["zcat", path(lane, "allSNP")], stdout=subprocess.PIPE)
        a = subprocess.Popen(["awk", "-v", f"FINE={C.FINE}", HIST_AWK],
                             stdin=z.stdout, stdout=subprocess.PIPE)
        z.stdout.close()
        parts.append(pd.read_csv(a.stdout, sep="\t", header=None,
                                 names=["chrom", "fine", "reads"]))
        a.wait(), z.wait()
        print(f"{lane}: {parts[-1].reads.sum()} reads", flush=True)
    h = pd.concat(parts).groupby(["chrom", "fine"], as_index=False).reads.sum()
    h.to_csv(out, index=False)
    return h


def lut():
    """FINE bin -> feature column, cutting N_BINS equal read mass bins per chromosome.

    the coordinate runs past the end of one chromosome into the next, so a FINE bin can
    straddle a boundary; those are dropped rather than assigned.
    """
    h = histogram()
    h = h[~h.fine.duplicated(keep=False) & h.chrom.isin(C.CHROMS)]
    rows = []
    for i, ch in enumerate(C.CHROMS):
        d = h[h.chrom == ch].sort_values("fine")
        r = d.reads.values
        mid = np.cumsum(r) - r / 2
        b = np.minimum((C.N_BINS * mid / r.sum()).astype(int), C.N_BINS - 1)
        rows.append(pd.DataFrame({"fine": d.fine.values, "feat": i * C.N_BINS + b}))
    return pd.concat(rows)


def lane(gem, bcs, lutfile):
    """(cells, N_FEAT, 2) binned G1/G2 counts for one lane, dense over bcs."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(bcs) + "\n")
        bcsfile = f.name
    pipe = (r"LC_ALL=C join -t $'\t' -1 2 -2 1 -o 1.2,1.1,1.3,2.2 "
            f"<(zcat {path(f'{C.TARGET}_{gem}', 'allSNP')}) "
            f"<(zcat {path(f'{C.TARGET}_{gem}', 'read.BC')})")
    j = subprocess.Popen(["bash", "-c", pipe], stdout=subprocess.PIPE)
    a = subprocess.Popen(["awk", "-v", f"FINE={C.FINE}", "-v", f"LUT={lutfile}",
                          "-v", f"BCS={bcsfile}", COUNT_AWK],
                         stdin=j.stdout, stdout=subprocess.PIPE)
    j.stdout.close()
    d = pd.read_csv(a.stdout, sep="\t", header=None, dtype=np.int32).values
    a.wait(), j.wait()
    os.unlink(bcsfile)
    key = d[:, 0].astype(np.int64) * (2 * C.N_FEAT) + d[:, 1]
    c = np.bincount(key, minlength=len(bcs) * 2 * C.N_FEAT)
    return c.reshape(len(bcs), C.N_FEAT, 2).astype(np.int32)


def main():
    """counts.npz: both lanes of the pool, stacked. autosomes only."""
    os.makedirs(C.OUT, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        lut().to_csv(f, sep="\t", header=False, index=False)
        lutfile = f.name
    ix = C.obs().index
    A, B, index = [], [], []
    for gem, sample in C.GEMS.items():
        suffix = f"-1-{sample}"
        bcs = sorted(b[:16] for b in ix if b.endswith(suffix))
        c = lane(gem, bcs, lutfile)
        A.append(c[:, :, 0])
        B.append(c[:, :, 1])
        index += [b + suffix for b in bcs]
        print(f"{gem}: {c.sum()} reads over {len(bcs)} cells", flush=True)
    os.unlink(lutfile)
    np.savez(C.out("counts.npz"), a=np.vstack(A), b=np.vstack(B),
             index=np.array(index), fine=C.FINE, n_bins=C.N_BINS)


if __name__ == "__main__":
    main()
