"""Unit tests for graph, grouping, scoring and queue logic (numpy / scipy / networkx only)."""
import unittest

import numpy as np

from flare import (StepGraph, cc_groups, ego_groups, flow_groups, group_features, group_score, pooled,
                        queue_stats, run_queue)
from flare.graph import same_address_pairs
from flare.groups import budget_for, select_seeds


def toy_graph():
    """Chain 0->1->2->3, fan-in 4->2, isolated 5, and a hub 6 linked to 7..40 (degree 34)."""
    src = [0, 1, 2, 4] + [6] * 34
    dst = [1, 2, 3, 2] + list(range(7, 41))
    return StepGraph(41, src, dst)


class TestStepGraph(unittest.TestCase):
    def test_neighbors_are_undirected_and_without_self_loops(self):
        G = StepGraph(3, [0, 1, 1], [1, 2, 1])
        self.assertEqual(sorted(G.neighbors(1).tolist()), [0, 2])
        self.assertEqual(G.degree.tolist(), [1, 2, 1])

    def test_components_restricted_to_node_subset(self):
        G = toy_graph()
        comps = sorted(sorted(c) for c in G.components([0, 1, 3, 4, 5]))
        self.assertEqual(comps, [[0, 1], [3], [4], [5]])

    def test_flow_degrees_count_only_internal_edges(self):
        G = toy_graph()
        indeg, outdeg, ne = G.flow_degrees([0, 1, 2, 4])
        self.assertEqual(ne, 3)
        self.assertEqual(indeg.tolist(), [0, 1, 2, 0])
        self.assertEqual(outdeg.tolist(), [1, 1, 0, 1])

    def test_same_address_pairs_chain_and_cap(self):
        addr = np.array([0, 0, 0, 1, 1, 2] + [3] * 12)
        step = np.array([1, 1, 1, 1, 2, 1] + [1] * 12)
        tx = np.array([5, 3, 4, 7, 8, 9] + list(range(20, 32)))
        a, b = same_address_pairs(addr, step, tx, address_cap=10)
        # address 0 in step 1 -> chain 3-4-5; address 1 spans two steps (no link); address 3 exceeds the cap
        self.assertEqual(sorted(zip(a.tolist(), b.tolist())), [(3, 4), (4, 5)])


class TestGroups(unittest.TestCase):
    def setUp(self):
        self.G = toy_graph()
        self.nodes = np.arange(41)
        self.risk = np.full(41, 0.01)
        self.risk[[0, 1, 2, 3, 4]] = [0.9, 0.8, 0.95, 0.7, 0.6]
        self.risk[6] = 0.99          # hub is the top seed
        self.risk[7:12] = 0.5

    def test_select_seeds_is_top_by_risk(self):
        s = select_seeds(self.nodes, self.risk, seed_frac=2 / 41)
        self.assertEqual(s.tolist(), [6, 2])

    def test_flow_groups_follow_high_risk_links_and_skip_hubs(self):
        groups = flow_groups(self.G, self.nodes, self.risk, seed_frac=2 / 41, q_exp=0.25, hops=2)
        as_sets = sorted(sorted(g.tolist()) for g in groups)
        # seed 2 expands through 1, 3, 4 and then 0; the hub seed 6 is kept alone (never expanded)
        self.assertIn([0, 1, 2, 3, 4], as_sets)
        self.assertIn([6], as_sets)
        for g in groups:
            self.assertFalse(set(g.tolist()) & set(range(7, 41)), "hub neighbours must not be pulled in")

    def test_hop_limit(self):
        groups = flow_groups(self.G, self.nodes, self.risk, seed_frac=1 / 41, q_exp=0.25, hops=1, hub_filter=False)
        # top seed 6 with hub filter off reaches its high-risk neighbours 7..11 in one hop
        self.assertEqual(sorted(groups[0].tolist()), [6, 7, 8, 9, 10, 11])

    def test_size_cap_splits_large_components(self):
        n = 60
        src = list(range(n - 1)); dst = list(range(1, n))   # a long chain
        G = StepGraph(n, src, dst)
        risk = np.linspace(1.0, 0.5, n)
        groups = flow_groups(G, np.arange(n), risk, seed_frac=0.5, q_exp=1.0, hops=5, max_size=20)
        self.assertGreater(len(groups), 1)
        self.assertTrue(all(set(g.tolist()) & set(range(30)) for g in groups), "each piece keeps a seed")

    def test_cc_groups_have_no_size_cap(self):
        groups = cc_groups(self.G, self.nodes, self.risk, seed_frac=1 / 41, q_exp=0.5)
        self.assertTrue(any(6 in g and len(g) > 5 for g in groups))

    def test_ego_groups_are_seed_plus_neighbours(self):
        groups = ego_groups(self.G, self.nodes, self.risk, seed_frac=2 / 41)
        self.assertEqual(sorted(groups[1].tolist()), [1, 2, 3, 4])


class TestScores(unittest.TestCase):
    def setUp(self):
        self.G = toy_graph()
        risk = np.zeros(41); risk[[0, 1, 2, 3, 4]] = [0.9, 0.8, 0.95, 0.7, 0.6]
        exposure = np.zeros(41); exposure[[2, 3]] = 1
        self.F = np.vstack([group_features(self.G, [0, 1, 2, 3, 4], risk, exposure),
                            group_features(self.G, [5], risk, exposure)])

    def test_feature_values(self):
        n, mx, mean, top3, s, motif, exp_ = self.F[0]
        self.assertEqual(n, 5)
        self.assertAlmostEqual(mx, 0.95)
        self.assertAlmostEqual(top3, (0.95 + 0.9 + 0.8) / 3)
        self.assertAlmostEqual(s, 3.95)
        self.assertAlmostEqual(motif, 2 / 5)   # node 1 passes funds through, node 2 merges (in-degree 2)
        self.assertAlmostEqual(exp_, 2 / 5)
        self.assertEqual(self.F[1, 5], 0.0)     # singleton has no motif

    def test_plain_scores_are_columns(self):
        for kind, col in [("max", 1), ("mean", 2), ("top3", 3), ("sum", 4)]:
            np.testing.assert_allclose(group_score(self.F, kind), self.F[:, col])

    def test_ours_reduces_to_sum_and_mean(self):
        np.testing.assert_allclose(group_score(self.F, "ours", (0, 0, 0, "sum")), self.F[:, 4])
        np.testing.assert_allclose(group_score(self.F, "ours", (0, 0, 1, "sum")), self.F[:, 2])

    def test_ours_rewards_motif_and_exposure(self):
        base = group_score(self.F, "ours", (0, 0, 0, "sum"))[0]
        boosted = group_score(self.F, "ours", (0.5, 0.5, 0, "sum"))[0]
        self.assertAlmostEqual(boosted, base * (1 + 0.5 * 0.4 + 0.5 * 0.4))

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            group_score(self.F, "median")


class TestQueues(unittest.TestCase):
    def setUp(self):
        self.G = toy_graph()
        self.nodes = np.arange(41)
        self.risk = np.linspace(1, 0, 41)
        self.y = np.full(41, -1); self.y[[0, 1, 2]] = 1; self.y[[3, 4]] = 0

    def test_budget_for(self):
        self.assertEqual(budget_for(4000, 0.01), 40)
        self.assertEqual(budget_for(10, 0.0001), 1)

    def test_queue_without_groups_is_top_risk(self):
        rev = run_queue([], np.array([]), self.risk, self.nodes, 5)
        self.assertEqual(rev.tolist(), [0, 1, 2, 3, 4])

    def test_queue_reviews_exactly_budget_without_duplicates(self):
        groups = [np.array([10, 11, 12]), np.array([11, 12, 13, 14, 15]), np.array([0])]
        rev = run_queue(groups, np.array([3.0, 2.0, 1.0]), self.risk, self.nodes, 6)
        self.assertEqual(len(rev), 6)
        self.assertEqual(len(set(rev.tolist())), 6)
        self.assertEqual(rev[:3].tolist(), [10, 11, 12])   # whole first group
        self.assertEqual(rev[3:].tolist(), [13, 14, 15])   # second group, new members only

    def test_overflowing_group_is_truncated_by_risk(self):
        rev = run_queue([np.array([20, 3, 30, 1])], np.array([1.0]), self.risk, self.nodes, 2)
        self.assertEqual(sorted(rev.tolist()), [1, 3])

    def test_queue_stats_and_pooling(self):
        rev = np.array([0, 1, 2, 3, 5])
        st = queue_stats(self.G, rev, self.y, self.nodes)
        # reviewed: 3 illicit, 1 licit, 1 unknown; cases: {0,1,2,3} and {5}
        self.assertEqual(st.tolist(), [3, 1, 1, 3, 2, 1, 5])
        p = pooled(np.vstack([st, st]))
        self.assertEqual(p["illicit_found"], 6)
        self.assertAlmostEqual(p["precision_known"], 0.75)
        self.assertAlmostEqual(p["cases_per_100"], 40.0)
        self.assertAlmostEqual(p["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
