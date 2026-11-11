# Results

Full run: Kaggle `onethybeing/btp-flowgroup-full`, 5 seeds, 30.9 min (T4 GPU). Split: train steps 1-30, validation 31-35, test 36-49.
Unknown labels are never used as negatives. History features use only labels from training steps and strictly earlier steps.
All tables: [`results/tables.md`](../results/tables.md), [`results/tables_extra.md`](../results/tables_extra.md) and the CSVs in [`results/`](../results/). All figures: [`figures/`](../figures/).

## Data audit (Elliptic++)
- 203,769 transactions, 234,355 flow edges, 822,942 addresses, 1,314,241 transaction-address links.
- **100% of transaction-flow edges connect transactions in the same time step.** So for the transaction graph, transductive and inductive message passing are equivalent.
- **The official wallet features are lifetime summaries.** For 100% of the 61,487 addresses that appear in more than one step, `total_txs` and `last_block_appeared_in` are identical at every step. 10.6% of address-step rows count steps that haven't happened yet. We therefore built point-in-time ("as-of") wallet features from the transaction-address links.

## E1 Transaction models (test PR-AUC, mean ± sd over 5 seeds)
| Model | Features | PR-AUC | Illicit F1 | F1 steps 36-42 | F1 steps 43-49 |
|---|---|---|---|---|---|
| LightGBM | AF+AUG+XV (as-of wallet view) | **0.750 ± 0.001** | 0.764 | 0.875 | 0.029 |
| LightGBM | AF+AUG | 0.747 ± 0.002 | 0.757 | 0.875 | 0.029 |
| LightGBM | AF+AUG+XV-lifetime (official) | 0.745 ± 0.001 | 0.752 | 0.874 | 0.027 |
| RF | AF+AUG+XV | 0.726 ± 0.002 | 0.773 | 0.877 | 0.041 |
| GraphSAGE (directed) | AF+AUG+XV | 0.720 ± 0.022 | 0.680 | 0.788 | 0.032 |
| GraphSAGE (undirected) | AF+AUG | 0.562 ± 0.043 | 0.537 | | |
| MLP | AF+AUG | 0.486 ± 0.045 | 0.530 | | |
| GraphSAGE (directed) | AF+AUG | 0.399 ± 0.056 | 0.453 | | |
| Logistic regression | AF+AUG+XV | 0.402 | 0.451 | | |

- Tree models beat GNNs, consistent with TWIN-GNN (2026).
- The as-of cross-view features give a small but consistent gain for LightGBM, and a large gain for GraphSAGE (0.40 → 0.72).
- Every model collapses after the dark-market shutdown at step 43 (F1 about 0.88 → 0.03).

## E2 Wallet models (address-level test PR-AUC)
| Features | PR-AUC |
|---|---|
| As-of wallet features | 0.273 |
| **As-of + transaction risk** | **0.645** |
| Official lifetime features | 0.345 |
| Official lifetime + transaction risk | 0.636 |

- Transaction-to-wallet risk transfer is the key cross-view signal (0.27 → 0.65).
- Lifetime features look better on their own (0.35 vs 0.27), partly because they contain future information, but add nothing once transaction risk is included.

## E3 Review queues (main result), 1% budget per step
| Risk model | Queue | Illicit found | Precision (known) | Cases per 100 reviewed | Illicit per case |
|---|---|---|---|---|---|
| LightGBM | Transaction queue | 208 | 0.99 | 83.7 | 0.40 |
| LightGBM | Wallet queue | 96 | 0.48 | 31.0 | 0.50 |
| LightGBM | **Flow groups (ours)** | **210** | 0.98 | **36.4** | **0.92** |
| LightGBM | Connected components | 205 | 0.96 | 32.4 | 1.01 |
| LightGBM | 1-hop neighbourhood | 167 | 0.77 | 26.7 | 1.00 |
| GraphSAGE | Transaction queue | 146 | 0.95 | 90.9 | 0.26 |
| GraphSAGE | **Flow groups (ours)** | **174** | 0.92 | **45.8** | **0.61** |
| GraphSAGE | Connected components | 181 | 0.92 | 42.0 | 0.69 |

Bootstrap 95% CIs over the 14 test steps, ours minus the transaction queue:
- **LightGBM risk.** Illicit found: +2 [-19, +26] at 1% (not significant) and +20 [+3, +41] at 5%. Cases: -296 [-400, -212] at 1% and -769 [-972, -627] at 5%.
- **GraphSAGE risk.** Illicit found: +29 [+2, +60] at 0.5%, +44 [+3, +90] at 2% and +22 [+1, +44] at 5%. Cases: always far fewer.
- **Against the wallet queue,** flow groups find significantly more illicit transactions at every budget (LightGBM risk).
- **Against the 1-hop neighbourhood,** flow groups find significantly more illicit transactions (e.g. +43 [+16, +76] at 1%).
- **Against connected components,** illicit found is not significantly different, but connected components give somewhat fewer cases.

## E4 Group proxy metrics (top-10 groups per test step)
- The top groups are 17-20 times more illicit-enriched than size-matched random connected groups, for every grouping method.
- About 38% of the top groups contain at least one illicit transaction (lenient hit) and about 33% contain at least two (strict).
- Flow groups are smaller than connected-component groups: max 21 vs 49 with LightGBM risk.

## E5 Counterfactual evidence test (GraphSAGE)
| Edges deleted (m) | Drop: gradient-top | Drop: top-risk heuristic | Drop: random | Drop: gradient-bottom | Top beats random |
|---|---|---|---|---|---|
| 1 | 0.256 | 0.305 | 0.069 | 0.029 | 59% of groups |
| 3 | 0.600 | 0.429 | 0.276 | 0.170 | 62% |
| 5 | 0.780 | 0.445 | 0.404 | 0.312 | 62% |

- The selected evidence edges matter about 2-4 times more than random edges.
- The gradient method beats the simple heuristic once m ≥ 3.

## X1 Supplementary: wider tuning grid (kernel `onethybeing/btp-flowgroup-extra`)
- The seed multiplier grid was widened to {1.0, 1.5, 2.0, 3.0}. Tuning now picks interior values: 2.0 for LightGBM and 1.5 for GraphSAGE. That resolves the earlier grid-edge issue.
- The conclusions are unchanged:
  - Groups match the transaction queue's yield with LightGBM risk (1%: 209 vs 208) and beat it with GraphSAGE risk (1%: 171 vs 147; 2%: 298 vs 260).
  - Groups give far fewer cases.
- **The validation-tuned extra score terms are unstable.** With LightGBM they gave more cases than the plain sum score (64.9 vs 41.2 per 100 at 1%) for similar yield. A plain sum-of-risk group score is the robust choice. Report it as the recommended configuration.

## X2 Supplementary: label scarcity (LightGBM risk, group configuration fixed from the full run, 3 random label subsets)
| Training labels | Transaction PR-AUC | Illicit @1%: transaction queue | Illicit @1%: flow groups | Illicit @1%: connected components | Cases/100: transaction queue | Cases/100: flow groups |
|---|---|---|---|---|---|---|
| 100% (26,905) | 0.750 | 207.7 ± 3.1 | 207.3 ± 5.0 | 203.3 ± 8.1 | 83.3 | 35.3 |
| 50% | 0.753 | 193.3 ± 14.0 | **212.0 ± 5.0** | 210.7 ± 4.2 | 84.3 | 34.9 |
| 20% | 0.732 | 170.3 ± 9.0 | **199.7 ± 5.9** | 203.3 ± 3.1 | 87.0 | 36.5 |
| 10% (2,690) | 0.722 | 175.3 ± 11.0 | **201.3 ± 11.5** | 203.3 ± 11.0 | 87.1 | 33.0 |

At the 2% budget: 10% labels give 292 (transaction queue) vs 320 (flow groups); 20% labels give 276 vs 317.

**Finding:**
- With fewer labels, transaction-level PR-AUC barely moves (0.75 → 0.72), but the transaction queue's top-1% yield falls about 16%.
- Group queues hold their yield (about 200–212) because they pull in the graph neighbours of high-risk seeds.
- So grouping gives **+15–17% more illicit transactions when only 10–20% of labels are available**, and still about 60% fewer cases.
- Connected components behave the same way: the benefit comes from structural grouping itself.

## Honest negatives and limitations
1. Our group-specific score terms (flow-pattern density, known-bad address exposure, size exponent) were selected on validation. On test they are almost identical to a plain sum-of-risk score (LightGBM 210 vs 202 illicit at 1%; GraphSAGE 174 vs 177).
2. The hub-filter and size-cap ablations change little: most groups are small (median 2-3).
3. Connected components on the risk-thresholded subgraph are an equally strong grouping baseline in yield.
4. No ring or campaign ground truth exists in Elliptic++. All group metrics are proxies.
5. Every model fails after step 43. That is concept drift, not a group-method issue.
6. Group parameters were tuned on only 5 validation steps, and the tuned extra score weights don't transfer reliably to test (see X1). The grid-edge issue was fixed in X1.
7. Group configurations were fixed (not re-tuned) in the label-scarcity study. Test-step budgets are small (about 16–70 transactions per step at 1%), so the counts have wide confidence intervals.
