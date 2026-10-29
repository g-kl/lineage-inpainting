from __future__ import annotations

import abc
import itertools
import logging
import numpy as np
from collections import deque, defaultdict

class RecordSampler(abc.ABC):
    """
    Strategy that jointly selects records from the pool and picks sampling time
    """

    @abc.abstractmethod
    def __call__(self, records: dict, n_records: int) -> tuple[list, float]:
        """
        Parameters
        ----------
        records   : full record dict (key -> record)
        n_records : how many records the trainer wants

        Returns
        -------
        chosen    : list of record objects (length == n_records)
        sample_time : float
        """
        ...


class PairedOffsetSampler(RecordSampler):
    """
    Default scheme for pairs of records (n_records == 2).

    Picks an anchor record at random, then finds a second record whose
    leaf_time is within ``tol`` of ``anchor.leaf_time + target_gap``.
    Anchors are restricted to records that have such a partner in the
    window.  The sampling time is ``min(leaf_time) - u`` over the chosen
    records, with ``u ~ U[0, sample_time_gap]``.

    For n_records > 2 the extra records are chosen uniformly at random
    from what remains and the sampling time is still taken from
    min(leaf_time) across all chosen records.

    Parameters
    ----------
    target_gap : float
        Desired difference in leaf_time between the two paired records.
    tol : float
        Half-width of the acceptance window around the target gap.
    sample_time_gap : float
        Maximum offset subtracted from ``min(leaf_time)`` to produce the
        sample time. The actual offset is drawn uniformly from
        ``[0, sample_time_gap]`` each call, so the lookup time spreads
        across all bins covered by the interval rather than always
        targeting a single bin.
    """

    def __init__(
        self,
        target_gap: float = 0.5,
        tol: float = 0.1,
        sample_time_gap: float = 0.0,
    ):
        self.target_gap = target_gap
        self.tol = tol
        self.sample_time_gap = sample_time_gap

    def __call__(self, records: dict, n_records: int) -> tuple[list, float]:
        keys = list(records.keys())
        # Restrict anchor candidates to records that have a valid forward
        # partner (another record within tol of anchor + target_gap). The
        # latest-leaf record has no forward partner; picking it as anchor
        # would leave _find_partner with no candidate.
        anchor_keys = [
            k for k in keys
            if any(
                k != k2
                and abs(records[k2].leaf_time
                        - (records[k].leaf_time + self.target_gap)) <= self.tol
                for k2 in keys
            )
        ]
        if not anchor_keys:
            raise RuntimeError(
                f"no record has a forward partner within tol={self.tol} of "
                f"target_gap={self.target_gap}; available leaf_times: "
                f"{sorted({r.leaf_time for r in records.values()})}"
            )
        anchor_key = anchor_keys[np.random.randint(len(anchor_keys))]
        anchor = records[anchor_key]
        anchor_time = anchor.leaf_time

        # Search for a partner ~target_gap away in leaf_time
        partner_key = self._find_partner(records, keys, anchor_key, anchor_time)
        chosen = [anchor, records[partner_key]]

        # Fill remaining slots (if n_records > 2) uniformly
        if n_records > 2:
            remaining = [k for k in keys if k not in {anchor_key, partner_key}]
            n_extra = min(n_records - 2, len(remaining))
            extra_idx = np.random.choice(len(remaining), size=n_extra, replace=False)
            chosen.extend(records[remaining[i]] for i in extra_idx)

        gap = float(np.random.uniform(0.0, self.sample_time_gap))
        sample_time = min(r.leaf_time for r in chosen) - gap
        return chosen, sample_time

    def _find_partner(self, records, keys, anchor_key, anchor_time) -> str:
        target = anchor_time + self.target_gap
        candidates = [
            k for k in keys
            if k != anchor_key
            and abs(records[k].leaf_time - target) <= self.tol
        ]
        return candidates[np.random.randint(len(candidates))]


class PairedOffsetBackfillSampler(PairedOffsetSampler):
    """``PairedOffsetSampler`` + one extra ``backfill`` record whose
    ``leaf_time`` sits ``backfill_gap`` (default 0.5) before the earliest
    paired record. Returned order is ``[backfill, anchor, partner]`` so the
    sequence is sorted by leaf_time.

    ``sample_time`` is anchored on the backfill record's ``leaf_time``:
    ``sample_time = backfill.leaf_time - uniform(0, sample_time_gap - backfill_gap)``.
    This guarantees every chosen record (backfill + paired) has structure at
    the lookup time, while still leaving the backfill with just a thin layer
    of leaves between ``sample_time`` and its ``leaf_time`` that the extender
    then grafts forward to the common target_leaf_time.

    If no record matches the backfill window the sampler falls back to the
    plain paired result; if ``n_records > 3`` extra records are filled in
    uniformly (matching the parent class's behavior).
    """

    def __init__(
        self,
        target_gap: float = 0.5,
        tol: float = 0.1,
        sample_time_gap: float = 0.6,
        backfill_gap: float = 0.5,
        backfill_tol: float = 0.1,
    ):
        super().__init__(
            target_gap=target_gap, tol=tol, sample_time_gap=sample_time_gap,
        )
        self.backfill_gap = float(backfill_gap)
        self.backfill_tol = float(backfill_tol)

    def __call__(self, records: dict, n_records: int) -> tuple[list, float]:
        # 1) plain paired sample (anchor + partner); we recompute sample_time below
        paired, _ = super().__call__(records, 2)
        # 2) find a backfill record at min(paired.leaf_time) - backfill_gap
        min_leaf = min(r.leaf_time for r in paired)
        target = min_leaf - self.backfill_gap
        chosen_ids = {id(r) for r in paired}
        cands = [
            k for k, r in records.items()
            if id(r) not in chosen_ids
            and abs(r.leaf_time - target) <= self.backfill_tol
        ]
        if cands:
            backfill_key = cands[np.random.randint(len(cands))]
        else:
            others = [k for k, r in records.items() if id(r) not in chosen_ids]
            if not others:
                # no backfill candidate at all: fall back to paired-only behavior
                gap = float(np.random.uniform(0.0, self.sample_time_gap))
                return paired, min_leaf - gap
            backfill_key = min(
                others, key=lambda k: abs(records[k].leaf_time - target),
            )
        backfill = records[backfill_key]
        # Anchor sample_time below the backfill record's leaf_time so every
        # chosen record has structure at the lookup time. The slack is the
        # extra width sample_time_gap allows past backfill_gap.
        slack = max(0.0, self.sample_time_gap - self.backfill_gap)
        gap = float(np.random.uniform(0.0, slack))
        sample_time = backfill.leaf_time - gap
        chosen = [backfill] + paired
        # 3) fill any extra slots beyond 3 uniformly
        if n_records > 3:
            taken = {id(r) for r in chosen}
            remaining = [k for k, r in records.items() if id(r) not in taken]
            n_extra = min(n_records - 3, len(remaining))
            if n_extra > 0:
                extra_idx = np.random.choice(
                    len(remaining), size=n_extra, replace=False,
                )
                chosen.extend(records[remaining[i]] for i in extra_idx)
        return chosen, sample_time


class AnchorTimeSampler(RecordSampler):
    """
    Samples a pair where one or both records have a specified leaf_time.

    ``start_anchor`` pins the smaller-leaf_time record; ``end_anchor`` pins the
    larger-leaf_time record. At least one must be provided.

    - If only ``start_anchor`` is set, the partner is chosen uniformly at random
      from records with leaf_time strictly greater than start_anchor.
    - If only ``end_anchor`` is set, the partner is chosen uniformly at random
      from records with leaf_time strictly less than end_anchor.
    - If both are set, one record is drawn for each anchor.

    Leaf-time comparisons use absolute tolerance ``tol`` to absorb floating
    point error. For n_records > 2 extra records are filled in uniformly at
    random. Sample time is ``min(leaf_time)`` across chosen records.
    """

    def __init__(
        self,
        start_anchor: float | None = None,
        end_anchor: float | None = None,
        tol: float = 0.1,
    ):
        if start_anchor is None and end_anchor is None:
            raise ValueError(
                "At least one of start_anchor or end_anchor must be provided"
            )
        self.start_anchor = start_anchor
        self.end_anchor = end_anchor
        self.tol = tol

    def __call__(self, records: dict, n_records: int) -> tuple[list, float]:
        keys = list(records.keys())

        if self.start_anchor is not None and self.end_anchor is not None:
            start_key = self._pick_anchor_key(records, keys, self.start_anchor)
            end_candidates = [
                k for k in keys
                if k != start_key
                and abs(records[k].leaf_time - self.end_anchor) <= self.tol
            ]
            if not end_candidates:
                raise ValueError(
                    f"No record found with leaf_time within {self.tol} of "
                    f"end_anchor {self.end_anchor} (excluding the start match)"
                )
            end_key = end_candidates[np.random.randint(len(end_candidates))]
            chosen_keys = [start_key, end_key]
        elif self.start_anchor is not None:
            start_key = self._pick_anchor_key(records, keys, self.start_anchor)
            partner_candidates = [
                k for k in keys
                if k != start_key
                and records[k].leaf_time > self.start_anchor + self.tol
            ]
            if not partner_candidates:
                raise ValueError(
                    f"No record with leaf_time strictly greater than "
                    f"start_anchor {self.start_anchor}"
                )
            partner_key = partner_candidates[np.random.randint(len(partner_candidates))]
            chosen_keys = [start_key, partner_key]
        else:
            end_key = self._pick_anchor_key(records, keys, self.end_anchor)
            partner_candidates = [
                k for k in keys
                if k != end_key
                and records[k].leaf_time < self.end_anchor - self.tol
            ]
            if not partner_candidates:
                raise ValueError(
                    f"No record with leaf_time strictly less than "
                    f"end_anchor {self.end_anchor}"
                )
            partner_key = partner_candidates[np.random.randint(len(partner_candidates))]
            chosen_keys = [partner_key, end_key]

        chosen = [records[k] for k in chosen_keys]

        if n_records > 2:
            taken = set(chosen_keys)
            remaining = [k for k in keys if k not in taken]
            n_extra = min(n_records - 2, len(remaining))
            extra_idx = np.random.choice(len(remaining), size=n_extra, replace=False)
            chosen.extend(records[remaining[i]] for i in extra_idx)

        sample_time = min(r.leaf_time for r in chosen)
        return chosen, sample_time

    def _pick_anchor_key(self, records, keys, anchor_time) -> str:
        matches = [
            k for k in keys
            if abs(records[k].leaf_time - anchor_time) <= self.tol
        ]
        if not matches:
            raise ValueError(
                f"No record found with leaf_time within {self.tol} of "
                f"anchor {anchor_time}"
            )
        return matches[np.random.randint(len(matches))]


class RandomTimeBinSampler(RecordSampler):
    """
    Picks ``n_records`` records uniformly at random, then chooses a
    sampling time by drawing uniformly from the time_bins (across all
    chosen records) that lie strictly below the minimum leaf_time.
    """

    def __call__(self, records: dict, n_records: int) -> tuple[list, float]:
        keys = list(records.keys())
        n_pick = min(n_records, len(keys))
        chosen_keys = np.random.choice(len(keys), size=n_pick, replace=False)
        chosen = [records[keys[i]] for i in chosen_keys]

        min_leaf = min(r.leaf_time for r in chosen)
        candidates = sorted({
            t for r in chosen for t in r.time_bins if t < min_leaf
        })
        if not candidates:
            raise ValueError(
                f"No time_bins below min leaf_time {min_leaf} across selected records"
            )
        sample_time = float(np.random.choice(candidates))
        return chosen, sample_time
