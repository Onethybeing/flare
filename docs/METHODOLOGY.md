# Methodology

## 1. Problem setting

Elliptic++ is a sequence of 49 time steps (about two weeks each). At step $t$ we observe

* transactions $V_t$ with feature vectors $x_i$ (93 local, 72 neighbourhood-aggregate and 17 augmented features),
* directed money-flow edges $E_t \subseteq V_t \times V_t$ (every flow edge joins two transactions of the same step),
* address-transaction links (an address spends into or receives from a transaction),
* labels $y_i \in \{1 \text{ (illicit)}, 0 \text{ (licit)}, -1 \text{ (unknown)}\}$; unknown is never treated as licit.

An analyst can review $B_t = \lceil b \cdot |V_t| \rceil$ transactions per step, for a budget fraction $b \in \{0.5\%, 1\%, 2\%, 5\%\}$.
The task is to choose *which* $B_t$ transactions to review so that (i) many known-illicit transactions
are found and (ii) the reviewed transactions form few, coherent cases.

**Splits.** Train on steps 1-30, select models and group parameters on 31-35, report test steps 36-49 once.
Thresholds, scalers, tuning grids and score weights never see test labels.

## 2. Point-in-time (as-of) address features

For address $a$ active at step $t$ (implemented in `flowgroups/asof.py`):

| Feature | Definition |
|---|---|
| `cur_send`, `cur_recv`, `cur_ntx`, `cur_btc`, `cur_fee` | activity of $a$ in step $t$ |
| `prior_cur_*` | the same quantities summed over steps $s < t$ |
| `prior_steps`, `age` | number of earlier active steps; $t - \text{first step of } a$ |
| `prior_ill`, `prior_lic` | labelled illicit / licit transactions of $a$ in steps $s < t$ **and** $s \le 30$ |
| `prior_ill_frac`, `prior_any_ill`, `prior_mean_btc` | ratios derived from the above |

Each transaction receives the mean and the maximum of these 15 features over its input addresses and over its
output addresses (60 columns, "XV"). The comparator "XV-lifetime" aggregates 13 official wallet features in the
same way. The known-bad *exposure* of a transaction is $\max$ of `prior_any_ill` over its addresses.

## 3. Risk models

* **LightGBM** (learning rate 0.03, 63 leaves, feature fraction 0.7, bagging 0.8, L2 = 1, early stopping on validation PR-AUC).
* **Random forest** (300 trees, 50 features per split) and **logistic regression** (balanced classes, standardised signed-log features).
* **Direction-aware GraphSAGE** (two layers, 128 units, dropout 0.3, weighted BCE with $\sqrt{n_-/n_+}$ positive weight):

$$h_v^{(l+1)} = \mathrm{ReLU}\Big(W^{(l)}\big[\,h_v^{(l)} \,\|\, \tfrac{\sum_{u\to v} w_{uv} h_u^{(l)}}{\sum_{u\to v} w_{uv}} \,\|\, \tfrac{\sum_{v\to u} w_{vu} h_u^{(l)}}{\sum_{v\to u} w_{vu}}\,\big]\Big)$$

  The undirected variant pools both directions; the MLP drops the neighbour terms. Because flow edges never cross
  time steps, full-graph training is equivalent to inductive, step-by-step message passing on this dataset.

Every stochastic model is trained with 5 seeds; queue experiments use the seed-averaged risk score.

## 4. Candidate groups

Implemented in `flowgroups/groups.py`; the step graph (`flowgroups/graph.py`) joins transactions by money flow
(either direction) and by shared addresses within the step (chained, only for address-step pairs with at most 10
transactions).

**Flow groups (proposed).** With seed fraction $\sigma$, expansion quantile $q$, hop limit $h$, hub degree $H=30$ and size cap $M=40$:

1. Seeds $S_t$: the $\max(1, \mathrm{round}(\sigma |V_t|))$ highest-risk transactions; $\sigma = \text{sfm} \times b$ scales with the budget.
2. Threshold $\tau_t$: the $(1-q)$ quantile of risk in step $t$.
3. From each non-hub seed, breadth-first expansion for at most $h$ hops, adding neighbour $u$ only if $r_u \ge \tau_t$ and $\deg(u) \le H$.
4. Connected components of the collected set that contain a seed are groups; components larger than $M$ are split with Louvain (seed 0), keeping pieces that contain a seed.

**Baselines.** *Connected components*: components of $\{v : r_v \ge \tau_t\} \cup S_t$ with no hop limit, hub filter or size cap.
*1-hop neighbourhood*: each seed plus all of its neighbours.

## 5. Group scores

For group $G$ with member risks $r_v$, the features are $n = |G|$, $\max$, mean, top-3 mean, sum,

$$\text{motif}(G) = \frac{|\{v: d^{out}_G(v) \ge 2\}| + |\{v: d^{in}_G(v) \ge 2\}| + |\{v: d^{in}_G(v) = d^{out}_G(v) = 1\}|}{n}$$

(split, merge and pass-through members inside the group's flow sub-graph) and exposure (mean known-bad indicator).
The proposed score is

$$S(G) = \mathrm{base}(G)\,\big(1 + \beta\,\text{motif}(G) + \gamma\,\text{exposure}(G)\big)\, n^{-\delta},\qquad \mathrm{base} \in \{\text{sum}, \text{top-3}\}$$

with $\delta = 0$ giving sum-like and $\delta = 1$ mean-like behaviour. The grid $\beta \in \{0, .25, .5, 1\}$,
$\gamma \in \{0, .5, 1\}$, $\delta \in \{0, .25, .5\}$ is searched jointly with $(\text{sfm}, q, h)$ on validation steps,
maximising illicit found summed over the four budgets. Plain max / mean / top-3 / sum scores are reported as ablations.

## 6. Review queues and metrics

`flowgroups/queues.py`: groups are reviewed whole in score order; the group that would overflow $B_t$ is truncated to
its highest-risk members; leftover capacity goes to the highest-risk unreviewed transactions. The transaction
queue is the special case with no groups; the wallet queue reviews addresses in wallet-model order, each covering
its transactions in the step.

| Metric | Definition (pooled over test steps) |
|---|---|
| Illicit found | $\sum_t |\{v \in R_t : y_v = 1\}|$ |
| Precision (known) | illicit / (illicit + licit) among reviewed |
| Recall | illicit found / illicit in the test steps |
| Label coverage | (illicit + licit) / reviewed |
| Cases per 100 | $100 \times \sum_t c(R_t) / \sum_t |R_t|$ where $c(R)$ is the number of connected components of the reviewed set |
| Illicit per case | illicit found / cases |

Node-level metrics: PR-AUC (primary; Saito and Rehmsmeier 2015), ROC-AUC, illicit F1 at the validation-optimal threshold,
P@100, P@500, and F1 before / after the step-43 dark-market shutdown.

**Uncertainty.** Differences between queues use a paired bootstrap over the 14 test steps (2,000 resamples); seeds give
mean ± sd for models.

## 7. Group proxy metrics

Elliptic++ has no ring or campaign identifiers, so group quality is measured by proxies on the top-10 groups per test
step: lenient hit (at least one illicit member), strict hit (at least two), labelled purity, label coverage, and
enrichment against size-matched random *connected* groups grown by BFS from random transactions in the same step (20 per group).

## 8. Counterfactual evidence test

For a top group of the GraphSAGE pipeline, all flow edges of its step get weight $w_e = 1$. The gradient
$\partial S / \partial w_e$ ranks edges; deleting the top-$m$ edges ($w_e = 0$) gives the drop $\Delta_{top}$, compared with
$m$ random edges (10 draws), the $m$ lowest-ranked edges and the $m$ edges touching the highest-risk member.
Sufficiency keeps only the top-$m$ edges inside the receptive field.

**Known property.** With normalised mean aggregation, an edge into a node that has only one neighbour in that direction
has zero gradient (the weight cancels in the mean), so such edges are never selected. The unit test
`test_sole_neighbour_edges_have_zero_saliency` documents this.

## 9. Supplementary experiments

* **X1:** seed-multiplier grid widened to {1.0, 1.5, 2.0, 3.0} (the main run's optimum, 1.5, sat at the grid edge).
* **X2:** label scarcity: training labels subsampled to 50/20/10% per class (3 random subsets); the as-of history labels use the
  same mask; group configurations are frozen from the full run.

## References

1. M. Weber *et al.*, "Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics," arXiv:1908.02591, 2019.
2. Y. Elmougy and L. Liu, "Demystifying fraudulent transactions and illicit nodes in the Bitcoin network for financial forensics," *Proc. KDD*, pp. 3979-3990, 2023, doi:10.1145/3580305.3599803.
3. A. Malik, "Do transaction-level and actor-level AML queues agree? An empirical evaluation of granularity effects on the Elliptic++ graph," arXiv:2604.23494, 2026.
4. A. P. Ichull and J. C. Sopuru, "TWIN-GNN: An interpretable multi-graph framework for illicit Bitcoin transaction and wallet detection," *J. Cybersecurity and Privacy*, vol. 6, no. 6, 186, 2026, doi:10.3390/jcp6060186.
5. N. Han, R. Zhang, X. Liu and H. Zhang, "Illicit Bitcoin transaction detection via feature-gated temporal graph learning," *Scientific Reports*, vol. 16, 22352, 2026, doi:10.1038/s41598-026-53783-y.
6. Y. Jiao *et al.*, "Dynamic heterogeneous graph contrastive learning for uncovering collusive financial fraud," *Scientific Reports*, vol. 16, 28993, 2026, doi:10.1038/s41598-026-58938-5.
7. W. L. Hamilton, R. Ying and J. Leskovec, "Inductive representation learning on large graphs," *NeurIPS*, 2017, arXiv:1706.02216.
8. G. Ke *et al.*, "LightGBM: A highly efficient gradient boosting decision tree," *NeurIPS*, 2017.
9. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5-32, 2001, doi:10.1023/A:1010933404324.
10. V. D. Blondel, J.-L. Guillaume, R. Lambiotte and E. Lefebvre, "Fast unfolding of communities in large networks," *J. Stat. Mech.*, P10008, 2008, doi:10.1088/1742-5468/2008/10/P10008.
11. S. Kaufman, S. Rosset, C. Perlich and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM TKDD*, vol. 6, no. 4, 2012, doi:10.1145/2382577.2382579.
12. T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, e0118432, 2015, doi:10.1371/journal.pone.0118432.
13. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. Chapman and Hall/CRC, 1994, doi:10.1201/9780429246593.
14. R. Ying *et al.*, "GNNExplainer: Generating explanations for graph neural networks," *NeurIPS*, 2019, arXiv:1903.03894.
15. Z. Zhang *et al.*, "ATM-GAD: Adaptive temporal motif graph anomaly detection for financial transaction networks," arXiv:2508.20829, 2025 (preprint).
