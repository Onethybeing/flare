"""Minimal end-to-end example on a synthetic transaction graph (runs in about a second).

    python examples/quickstart.py

It plants three small laundering-like flows in a random graph, scores transactions with a
noisy risk model, and compares a transaction queue with a flow-group queue at the same budget.
"""
import numpy as np

from flowgroups import StepGraph, flow_groups, group_features, group_score, pooled, queue_stats, run_queue
from flowgroups.groups import budget_for

rng = np.random.default_rng(0)
n = 2000

# background money-flow edges plus one exchange-like hub
src = list(rng.integers(0, n, 2500)); dst = list(rng.integers(0, n, 2500))
src += [0] * 120; dst += list(rng.integers(1, n, 120))

# three planted illicit flows (chains with a merge)
y = np.full(n, -1); y[rng.choice(n, 400, replace=False)] = 0
flows = [[100, 101, 102, 103, 104], [200, 201, 202, 203], [300, 301, 302, 303, 304, 305]]
for f in flows:
    for a, b in zip(f[:-1], f[1:]):
        src.append(a); dst.append(b)
    y[f] = 1
    y[f[1]] = -1                     # one member of each flow is unlabelled

G = StepGraph(n, src, dst)
risk = np.clip(rng.beta(1, 12, n) + 0.75 * (y == 1) + 0.35 * np.isin(np.arange(n), sum(flows, [])), 0, 1)
risk[0] = 0.9                        # the hub looks risky too

nodes = np.arange(n)
B = budget_for(n, 0.01)              # review 1% of the step = 20 transactions

tx_queue = run_queue([], np.array([]), risk, nodes, B)
groups = flow_groups(G, nodes, risk, seed_frac=0.01, q_exp=0.05, hops=2, hub_degree=30, max_size=40)
F = np.array([group_features(G, g, risk) for g in groups])
group_queue = run_queue(groups, group_score(F, "sum"), risk, nodes, B)

for name, rev in [("transaction queue", tx_queue), ("flow-group queue", group_queue)]:
    p = pooled(queue_stats(G, rev, y, nodes))
    print(f"{name:18s} reviewed={B}  illicit found={p['illicit_found']:2d}  "
          f"cases per 100 reviewed={p['cases_per_100']:5.1f}  illicit per case={p['illicit_per_case']:.2f}")
print(f"{len(groups)} candidate groups; largest has {max(len(g) for g in groups)} transactions")
