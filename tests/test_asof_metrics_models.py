"""Tests for point-in-time features, the leakage audit, bootstrap CIs and the GraphSAGE model."""
import unittest

import numpy as np
import pandas as pd

from flare.asof import address_step_table, aggregate_to_transactions, lifetime_leakage_audit
from flare.metrics import paired_bootstrap_ci

try:
    import torch
    from flare.models import DirSAGE
    from flare.counterfactual import deletion_test, edge_saliency
except ImportError:  # pragma: no cover
    torch = None


def toy_links():
    # address 0 appears at steps 1, 3, 40 (40 is after the training cut-off of 30)
    return pd.DataFrame({
        "a":    [0, 0, 0, 0, 1],
        "tx":   [10, 11, 12, 13, 14],
        "ts":   [1, 3, 3, 40, 3],
        "role": [0, 1, 0, 1, 1],
        "ylab": [1, 0, -1, 1, 1],
        "btc":  [1.0, 2.0, 3.0, 4.0, 5.0],
        "fee":  [0.1, 0.1, 0.1, 0.1, 0.1],
    })


class TestAsOf(unittest.TestCase):
    def setUp(self):
        self.A = address_step_table(toy_links(), train_end=30).set_index(["a", "ts"])

    def test_first_appearance_has_no_history(self):
        r = self.A.loc[(0, 1)]
        self.assertEqual(r.prior_cur_ntx, 0)
        self.assertEqual(r.prior_ill, 0)
        self.assertEqual(r.age, 0)

    def test_history_uses_only_strictly_earlier_steps(self):
        r = self.A.loc[(0, 3)]
        self.assertEqual(r.cur_ntx, 2)            # two txs in step 3
        self.assertEqual(r.prior_cur_ntx, 1)      # only the step-1 tx counts as history
        self.assertEqual(r.prior_ill, 1)          # step-1 illicit label is visible
        self.assertEqual(r.prior_lic, 0)          # step-3 licit label is NOT visible at step 3
        self.assertEqual(r.age, 2)

    def test_labels_after_training_period_never_enter_history(self):
        r = self.A.loc[(0, 40)]
        self.assertEqual(r.prior_ill, 1)          # step-1 illicit only; step-40 label excluded
        self.assertEqual(r.prior_lic, 1)          # step-3 licit (training period, earlier step)
        self.assertEqual(r.prior_steps, 2)

    def test_aggregate_to_transactions_shapes(self):
        links = toy_links()
        A = address_step_table(links, train_end=30)
        X = aggregate_to_transactions(links, A, n_tx=15)
        self.assertEqual(X.shape[0], 15)
        self.assertTrue(np.isnan(X.loc[0]).all())               # tx 0 has no links
        self.assertEqual(X.loc[11, "xv_out_prior_ill_max"], 1)  # receiver address had an illicit past


class TestLeakageAudit(unittest.TestCase):
    def test_detects_lifetime_features(self):
        links = toy_links()
        A = address_step_table(links, train_end=30)
        wallet = pd.DataFrame({"a": [0, 0, 0], "ts": [1, 3, 40], "total_txs": [4, 4, 4],
                               "last_block_appeared_in": [99, 99, 99], "num_timesteps_appeared_in": [3, 3, 3]})
        audit = lifetime_leakage_audit(wallet, A)
        self.assertEqual(audit["frac_multi_step_addr_constant_total_txs"], 1.0)
        # rows at steps 1 and 3 already "know" about 3 active steps -> future information
        self.assertAlmostEqual(audit["frac_rows_counting_future_steps"], 2 / 3)


class TestBootstrap(unittest.TestCase):
    def test_identical_queues_give_zero_interval(self):
        a = np.array([3, 5, 2, 8])
        self.assertEqual(paired_bootstrap_ci(a, a), (0.0, 0.0))

    def test_interval_contains_true_difference(self):
        rng = np.random.default_rng(0)
        b = rng.integers(0, 20, 50)
        a = b + 2
        lo, hi = paired_bootstrap_ci(a, b)
        self.assertEqual((lo, hi), (100.0, 100.0))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            paired_bootstrap_ci([1, 2], [1, 2, 3])


@unittest.skipIf(torch is None, "PyTorch not installed")
class TestDirSAGE(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(6, 4)
        self.src = torch.tensor([0, 1, 2, 3])
        self.dst = torch.tensor([1, 2, 3, 4])
        self.model = DirSAGE(4, hidden=8, dropout=0.0).eval()

    def test_output_shape(self):
        self.assertEqual(self.model(self.x, self.src, self.dst).shape, (6,))

    def test_zero_weight_equals_edge_removal(self):
        w = torch.ones(4); w[1] = 0
        with torch.no_grad():
            a = self.model(self.x, self.src, self.dst, w)
            b = self.model(self.x, self.src[[0, 2, 3]], self.dst[[0, 2, 3]])
        torch.testing.assert_close(a, b)

    def test_direction_matters(self):
        with torch.no_grad():
            a = self.model(self.x, self.src, self.dst)
            b = self.model(self.x, self.dst, self.src)
        self.assertFalse(torch.allclose(a, b))

    def test_mlp_ignores_edges(self):
        m = DirSAGE(4, hidden=8, graph=False, dropout=0.0).eval()
        with torch.no_grad():
            torch.testing.assert_close(m(self.x, self.src, self.dst), m(self.x, self.src[:1], self.dst[:1]))

    def test_isolated_node_unaffected_by_edges(self):
        with torch.no_grad():
            a = self.model(self.x, self.src, self.dst)[5]
            b = self.model(self.x, self.src[:2], self.dst[:2])[5]
        torch.testing.assert_close(a, b)

    def test_sole_neighbour_edges_have_zero_saliency(self):
        # mean aggregation normalises by the weight sum, so an edge into a node with a single
        # neighbour in that direction cancels out: documented limitation of gradient saliency
        s, g = edge_saliency(self.model, self.x, self.src, self.dst, torch.tensor([2, 3]))
        self.assertTrue(np.allclose(g, 0))

    def test_counterfactual_deletion_on_fan_in(self):
        torch.manual_seed(1)
        model = DirSAGE(4, hidden=64, dropout=0.0).eval()
        src = torch.tensor([0, 1, 4, 5, 2])      # 0, 1, 4, 5 all send to 2; 2 sends to 3
        dst = torch.tensor([2, 2, 2, 2, 3])
        s, g = edge_saliency(model, self.x, src, dst, torch.tensor([2, 3]))
        self.assertGreaterEqual(int((np.abs(g) > 0).sum()), 2)
        res = deletion_test(model, self.x, src, dst, torch.tensor([2, 3]), m=1)
        self.assertIsNotNone(res)
        self.assertGreaterEqual(res["d_top"], res["d_bottom"])


if __name__ == "__main__":
    unittest.main()
