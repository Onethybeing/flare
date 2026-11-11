"""Fixed-budget review queues and case-fragmentation statistics.

A queue reviews exactly ``B`` transactions of a time step. Groups are taken in score order and
reviewed whole; the group that would overflow the budget is truncated to its highest-risk
members; if groups run out, the remaining capacity is filled with the highest-risk unreviewed
transactions. With no groups the queue is the plain transaction queue (top-``B`` by risk).

Statistics per queue (``QUEUE_STAT_NAMES``):

* ``ill, lic, unk``: illicit, licit and unlabelled transactions reviewed (unknown is *not* licit)
* ``tot_ill``: illicit transactions in the step (recall denominator)
* ``cases``: connected components among the reviewed transactions (fragmentation)
* ``ill_cases``: cases containing at least one illicit transaction
* ``reviewed``: transactions reviewed
"""
from __future__ import annotations

import numpy as np

from .graph import StepGraph

QUEUE_STAT_NAMES = ["ill", "lic", "unk", "tot_ill", "cases", "ill_cases", "reviewed"]


def run_queue(groups, scores, risk, nodes, budget: int) -> np.ndarray:
    """Return the reviewed transaction ids (length ``min(budget, len(nodes))``, no duplicates)."""
    nodes = np.asarray(nodes)
    scores = np.asarray(scores, dtype=float)
    seen: set[int] = set()
    reviewed: list[int] = []
    for gi in np.argsort(-scores, kind="stable"):
        new = [int(v) for v in groups[gi] if int(v) not in seen]
        if not new:
            continue
        if len(reviewed) + len(new) > budget:
            new = sorted(new, key=lambda v: -risk[v])[: budget - len(reviewed)]
        reviewed += new
        seen.update(new)
        if len(reviewed) >= budget:
            break
    if len(reviewed) < budget:
        for v in nodes[np.argsort(-risk[nodes], kind="stable")]:
            v = int(v)
            if v not in seen:
                reviewed.append(v)
                seen.add(v)
            if len(reviewed) >= budget:
                break
    return np.array(reviewed, dtype=np.int64)


def queue_stats(G: StepGraph, reviewed, y, nodes) -> np.ndarray:
    """Statistics vector in ``QUEUE_STAT_NAMES`` order. Labels: 1 illicit, 0 licit, -1 unknown."""
    reviewed = np.asarray(reviewed, dtype=np.int64)
    yy = y[reviewed]
    comps = G.components(reviewed)
    return np.array([(yy == 1).sum(), (yy == 0).sum(), (yy == -1).sum(), (y[np.asarray(nodes)] == 1).sum(),
                     len(comps), sum(1 for c in comps if (y[np.asarray(c)] == 1).any()), len(reviewed)])


def pooled(stats: np.ndarray) -> dict:
    """Pool per-step statistics (rows) into the headline queue metrics."""
    s = dict(zip(QUEUE_STAT_NAMES, np.atleast_2d(stats).sum(0)))
    return dict(
        illicit_found=int(s["ill"]),
        precision_known=s["ill"] / max(s["ill"] + s["lic"], 1),
        recall=s["ill"] / max(s["tot_ill"], 1),
        label_coverage=(s["ill"] + s["lic"]) / max(s["reviewed"], 1),
        cases_per_100=100 * s["cases"] / max(s["reviewed"], 1),
        illicit_per_case=s["ill"] / max(s["cases"], 1),
        illicit_cases=int(s["ill_cases"]),
    )
