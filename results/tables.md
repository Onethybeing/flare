# Benchmark tables

## E1 Transaction-level models (test steps 36-49, mean over seeds)

| model | features | AP_mean | AP_std | AUC_mean | F1_mean | F1_std | P@100_mean | P@500_mean | F1_pre43_mean | F1_post43_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | AF+AUG+XV | 0.75 | 0.001 | 0.924 | 0.764 | 0.008 | 1.0 | 0.994 | 0.875 | 0.029 |
| LightGBM | AF+AUG | 0.747 | 0.002 | 0.919 | 0.757 | 0.012 | 1.0 | 0.994 | 0.875 | 0.029 |
| LightGBM | AF+AUG+XVlifetime | 0.745 | 0.001 | 0.916 | 0.752 | 0.016 | 1.0 | 0.998 | 0.874 | 0.027 |
| LightGBM | AF | 0.745 | 0.001 | 0.914 | 0.762 | 0.007 | 1.0 | 0.994 | 0.879 | 0.028 |
| RF | AF+AUG+XV | 0.726 | 0.002 | 0.898 | 0.773 | 0.011 | 1.0 | 0.962 | 0.877 | 0.041 |
| LightGBM | LF | 0.725 | 0.001 | 0.9 | 0.728 | 0.006 | 1.0 | 0.992 | 0.836 | 0.022 |
| RF | AF | 0.725 | 0.001 | 0.884 | 0.7 | 0.04 | 1.0 | 1.0 | 0.842 | 0.02 |
| RF | AF+AUG | 0.723 | 0.001 | 0.88 | 0.724 | 0.012 | 1.0 | 1.0 | 0.852 | 0.023 |
| SAGE-dir | AF+AUG+XV | 0.72 | 0.022 | 0.938 | 0.68 | 0.035 | 1.0 | 0.937 | 0.788 | 0.032 |
| SAGE-undir | AF+AUG | 0.562 | 0.043 | 0.895 | 0.537 | 0.039 | 0.962 | 0.73 | 0.633 | 0.013 |
| MLP | AF+AUG | 0.486 | 0.045 | 0.893 | 0.53 | 0.011 | 0.776 | 0.644 | 0.615 | 0.018 |
| LogReg | AF+AUG+XV | 0.402 | nan | 0.881 | 0.451 | nan | 0.57 | 0.59 | 0.606 | 0.037 |
| SAGE-dir | AF+AUG | 0.399 | 0.056 | 0.888 | 0.453 | 0.01 | 0.742 | 0.47 | 0.555 | 0.012 |
| LogReg | AF+AUG | 0.276 | nan | 0.874 | 0.341 | nan | 0.2 | 0.344 | 0.469 | 0.03 |
| LogReg | AF | 0.256 | nan | 0.866 | 0.341 | nan | 0.07 | 0.324 | 0.474 | 0.023 |

## E2 Wallet-level models

| features | row_AP_mean | row_AP_std | row_F1_mean | row_F1_std | addr_AP_mean | addr_AP_std | newaddr_AP_mean | newaddr_AP_std | n_new_addr_mean | n_new_addr_std |
|---|---|---|---|---|---|---|---|---|---|---|
| W-asof | 0.271 | 0.029 | 0.339 | 0.034 | 0.273 | 0.03 | 0.268 | 0.029 | 87279.0 | 0.0 |
| W-asof+txrisk | 0.638 | 0.002 | 0.627 | 0.015 | 0.645 | 0.002 | 0.644 | 0.002 | 87279.0 | 0.0 |
| W-official(lifetime) | 0.34 | 0.017 | 0.375 | 0.034 | 0.345 | 0.018 | 0.343 | 0.019 | 87279.0 | 0.0 |
| W-official+txrisk(lifetime) | 0.628 | 0.004 | 0.638 | 0.005 | 0.636 | 0.003 | 0.638 | 0.003 | 87279.0 | 0.0 |

## Leakage audit

| rows | addresses_multi_step | frac_multi_step_addr_constant_total_txs | frac_multi_step_addr_constant_last_block | frac_rows_counting_future_steps |
|---|---|---|---|---|
| 920691.0 | 61487.0 | 1.0 | 1.0 | 0.106 |

## E3 Review queues at budget 0.5% per step

| risk | queue | budget | illicit_found | precision_known | recall | label_coverage | cases_per_100 | illicit_per_case | illicit_cases |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | tx-queue (no groups) | 0.005 | 115 | 0.983 | 0.128 | 0.371 | 81.905 | 0.446 | 85 |
| LightGBM | wallet-queue | 0.005 | 52 | 0.419 | 0.058 | 0.394 | 33.651 | 0.491 | 34 |
| LightGBM | flow(ours) | ours | 0.005 | 108 | 0.956 | 0.12 | 0.359 | 37.143 | 0.923 | 20 |
| LightGBM | conncomp | sum | 0.005 | 107 | 0.973 | 0.119 | 0.349 | 43.175 | 0.787 | 44 |
| LightGBM | ego1hop | mean | 0.005 | 95 | 0.812 | 0.105 | 0.371 | 28.889 | 1.044 | 33 |
| LightGBM | flow(ours) | max (ablation: plain score) | 0.005 | 106 | 1.0 | 0.118 | 0.337 | 47.937 | 0.702 | 38 |
| LightGBM | flow(ours) | mean (ablation: plain score) | 0.005 | 108 | 0.991 | 0.12 | 0.346 | 78.095 | 0.439 | 67 |
| LightGBM | flow(ours) | top3 (ablation: plain score) | 0.005 | 110 | 0.973 | 0.122 | 0.359 | 58.095 | 0.601 | 44 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.005 | 107 | 0.955 | 0.119 | 0.356 | 36.19 | 0.939 | 20 |
| LightGBM | flow(ours) | no hub filter (ablation) | 0.005 | 108 | 0.956 | 0.12 | 0.359 | 37.143 | 0.923 | 20 |
| LightGBM | flow(ours) | no size cap (ablation) | 0.005 | 108 | 0.956 | 0.12 | 0.359 | 37.143 | 0.923 | 20 |
| GraphSAGE | tx-queue (no groups) | 0.005 | 74 | 0.961 | 0.082 | 0.244 | 94.921 | 0.247 | 68 |
| GraphSAGE | wallet-queue | 0.005 | 52 | 0.419 | 0.058 | 0.394 | 33.651 | 0.491 | 34 |
| GraphSAGE | flow(ours) | ours | 0.005 | 103 | 0.945 | 0.114 | 0.346 | 46.032 | 0.71 | 24 |
| GraphSAGE | conncomp | sum | 0.005 | 95 | 0.922 | 0.105 | 0.327 | 41.905 | 0.72 | 20 |
| GraphSAGE | ego1hop | mean | 0.005 | 87 | 0.806 | 0.097 | 0.343 | 30.476 | 0.906 | 19 |
| GraphSAGE | flow(ours) | max (ablation: plain score) | 0.005 | 76 | 0.95 | 0.084 | 0.254 | 64.762 | 0.373 | 21 |
| GraphSAGE | flow(ours) | mean (ablation: plain score) | 0.005 | 65 | 0.985 | 0.072 | 0.21 | 92.698 | 0.223 | 58 |
| GraphSAGE | flow(ours) | top3 (ablation: plain score) | 0.005 | 74 | 0.987 | 0.082 | 0.238 | 76.825 | 0.306 | 26 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.005 | 103 | 0.945 | 0.114 | 0.346 | 46.032 | 0.71 | 24 |
| GraphSAGE | flow(ours) | no hub filter (ablation) | 0.005 | 104 | 0.945 | 0.115 | 0.349 | 45.714 | 0.722 | 25 |
| GraphSAGE | flow(ours) | no size cap (ablation) | 0.005 | 103 | 0.945 | 0.114 | 0.346 | 46.032 | 0.71 | 24 |

## E3 Review queues at budget 1% per step

| risk | queue | budget | illicit_found | precision_known | recall | label_coverage | cases_per_100 | illicit_per_case | illicit_cases |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | tx-queue (no groups) | 0.01 | 208 | 0.986 | 0.231 | 0.337 | 83.706 | 0.397 | 152 |
| LightGBM | wallet-queue | 0.01 | 96 | 0.478 | 0.107 | 0.321 | 30.99 | 0.495 | 58 |
| LightGBM | flow(ours) | ours | 0.01 | 210 | 0.981 | 0.233 | 0.342 | 36.422 | 0.921 | 41 |
| LightGBM | conncomp | sum | 0.01 | 205 | 0.958 | 0.228 | 0.342 | 32.428 | 1.01 | 35 |
| LightGBM | ego1hop | mean | 0.01 | 167 | 0.773 | 0.185 | 0.345 | 26.677 | 1.0 | 49 |
| LightGBM | flow(ours) | max (ablation: plain score) | 0.01 | 208 | 0.986 | 0.231 | 0.337 | 48.562 | 0.684 | 67 |
| LightGBM | flow(ours) | mean (ablation: plain score) | 0.01 | 160 | 0.994 | 0.178 | 0.257 | 83.866 | 0.305 | 121 |
| LightGBM | flow(ours) | top3 (ablation: plain score) | 0.01 | 206 | 0.986 | 0.229 | 0.334 | 59.425 | 0.554 | 70 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.01 | 202 | 0.962 | 0.224 | 0.335 | 35.144 | 0.918 | 39 |
| LightGBM | flow(ours) | no hub filter (ablation) | 0.01 | 210 | 0.981 | 0.233 | 0.342 | 35.623 | 0.942 | 41 |
| LightGBM | flow(ours) | no size cap (ablation) | 0.01 | 209 | 0.981 | 0.232 | 0.34 | 35.942 | 0.929 | 38 |
| GraphSAGE | tx-queue (no groups) | 0.01 | 146 | 0.948 | 0.162 | 0.246 | 90.895 | 0.257 | 121 |
| GraphSAGE | wallet-queue | 0.01 | 95 | 0.475 | 0.105 | 0.319 | 30.99 | 0.49 | 58 |
| GraphSAGE | flow(ours) | ours | 0.01 | 174 | 0.916 | 0.193 | 0.304 | 45.847 | 0.606 | 55 |
| GraphSAGE | conncomp | sum | 0.01 | 181 | 0.919 | 0.201 | 0.315 | 42.013 | 0.688 | 51 |
| GraphSAGE | ego1hop | mean | 0.01 | 158 | 0.76 | 0.175 | 0.332 | 28.594 | 0.883 | 38 |
| GraphSAGE | flow(ours) | max (ablation: plain score) | 0.01 | 160 | 0.941 | 0.178 | 0.272 | 63.898 | 0.4 | 66 |
| GraphSAGE | flow(ours) | mean (ablation: plain score) | 0.01 | 121 | 0.953 | 0.134 | 0.203 | 90.096 | 0.215 | 100 |
| GraphSAGE | flow(ours) | top3 (ablation: plain score) | 0.01 | 161 | 0.953 | 0.179 | 0.27 | 72.524 | 0.355 | 76 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.01 | 177 | 0.917 | 0.196 | 0.308 | 45.687 | 0.619 | 56 |
| GraphSAGE | flow(ours) | no hub filter (ablation) | 0.01 | 177 | 0.922 | 0.196 | 0.307 | 44.888 | 0.63 | 57 |
| GraphSAGE | flow(ours) | no size cap (ablation) | 0.01 | 174 | 0.916 | 0.193 | 0.304 | 45.847 | 0.606 | 55 |

## E3 Review queues at budget 2% per step

| risk | queue | budget | illicit_found | precision_known | recall | label_coverage | cases_per_100 | illicit_per_case | illicit_cases |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | tx-queue (no groups) | 0.02 | 326 | 0.985 | 0.362 | 0.265 | 84.683 | 0.309 | 244 |
| LightGBM | wallet-queue | 0.02 | 171 | 0.486 | 0.19 | 0.282 | 26.945 | 0.509 | 89 |
| LightGBM | flow(ours) | ours | 0.02 | 338 | 0.952 | 0.375 | 0.285 | 36.969 | 0.733 | 83 |
| LightGBM | conncomp | sum | 0.02 | 334 | 0.949 | 0.371 | 0.282 | 27.666 | 0.968 | 73 |
| LightGBM | ego1hop | mean | 0.02 | 268 | 0.673 | 0.297 | 0.319 | 24.779 | 0.867 | 72 |
| LightGBM | flow(ours) | max (ablation: plain score) | 0.02 | 363 | 0.968 | 0.403 | 0.301 | 55.253 | 0.527 | 142 |
| LightGBM | flow(ours) | mean (ablation: plain score) | 0.02 | 223 | 0.97 | 0.248 | 0.184 | 83.4 | 0.214 | 169 |
| LightGBM | flow(ours) | top3 (ablation: plain score) | 0.02 | 347 | 0.964 | 0.385 | 0.289 | 66.079 | 0.421 | 151 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.02 | 337 | 0.952 | 0.374 | 0.284 | 33.44 | 0.808 | 82 |
| LightGBM | flow(ours) | no hub filter (ablation) | 0.02 | 335 | 0.952 | 0.372 | 0.282 | 34.483 | 0.779 | 81 |
| LightGBM | flow(ours) | no size cap (ablation) | 0.02 | 342 | 0.953 | 0.38 | 0.288 | 36.808 | 0.745 | 83 |
| GraphSAGE | tx-queue (no groups) | 0.02 | 266 | 0.943 | 0.295 | 0.226 | 88.212 | 0.242 | 209 |
| GraphSAGE | wallet-queue | 0.02 | 171 | 0.486 | 0.19 | 0.282 | 26.945 | 0.509 | 89 |
| GraphSAGE | flow(ours) | ours | 0.02 | 310 | 0.893 | 0.344 | 0.278 | 57.017 | 0.436 | 136 |
| GraphSAGE | conncomp | sum | 0.02 | 312 | 0.897 | 0.346 | 0.279 | 55.734 | 0.449 | 133 |
| GraphSAGE | ego1hop | mean | 0.02 | 252 | 0.712 | 0.28 | 0.284 | 26.383 | 0.766 | 55 |
| GraphSAGE | flow(ours) | max (ablation: plain score) | 0.02 | 283 | 0.922 | 0.314 | 0.246 | 68.805 | 0.33 | 149 |
| GraphSAGE | flow(ours) | mean (ablation: plain score) | 0.02 | 220 | 0.924 | 0.244 | 0.191 | 83.962 | 0.21 | 165 |
| GraphSAGE | flow(ours) | top3 (ablation: plain score) | 0.02 | 270 | 0.928 | 0.3 | 0.233 | 76.905 | 0.282 | 155 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.02 | 312 | 0.897 | 0.346 | 0.279 | 57.017 | 0.439 | 138 |
| GraphSAGE | flow(ours) | no hub filter (ablation) | 0.02 | 310 | 0.893 | 0.344 | 0.278 | 56.696 | 0.438 | 136 |
| GraphSAGE | flow(ours) | no size cap (ablation) | 0.02 | 310 | 0.893 | 0.344 | 0.278 | 57.017 | 0.436 | 136 |

## E3 Review queues at budget 5% per step

| risk | queue | budget | illicit_found | precision_known | recall | label_coverage | cases_per_100 | illicit_per_case | illicit_cases |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | tx-queue (no groups) | 0.05 | 527 | 0.923 | 0.585 | 0.184 | 79.588 | 0.213 | 342 |
| LightGBM | wallet-queue | 0.05 | 372 | 0.439 | 0.413 | 0.273 | 21.024 | 0.57 | 124 |
| LightGBM | flow(ours) | ours | 0.05 | 547 | 0.897 | 0.607 | 0.196 | 54.829 | 0.321 | 250 |
| LightGBM | conncomp | sum | 0.05 | 548 | 0.894 | 0.608 | 0.197 | 51.642 | 0.342 | 251 |
| LightGBM | ego1hop | mean | 0.05 | 421 | 0.515 | 0.467 | 0.263 | 19.478 | 0.696 | 56 |
| LightGBM | flow(ours) | max (ablation: plain score) | 0.05 | 557 | 0.938 | 0.618 | 0.191 | 60.947 | 0.294 | 265 |
| LightGBM | flow(ours) | mean (ablation: plain score) | 0.05 | 465 | 0.914 | 0.516 | 0.164 | 72.44 | 0.207 | 256 |
| LightGBM | flow(ours) | top3 (ablation: plain score) | 0.05 | 518 | 0.923 | 0.575 | 0.181 | 67.16 | 0.248 | 254 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.05 | 548 | 0.894 | 0.608 | 0.197 | 52.254 | 0.338 | 251 |
| LightGBM | flow(ours) | no hub filter (ablation) | 0.05 | 547 | 0.897 | 0.607 | 0.196 | 54.475 | 0.323 | 250 |
| LightGBM | flow(ours) | no size cap (ablation) | 0.05 | 547 | 0.897 | 0.607 | 0.196 | 54.829 | 0.321 | 250 |
| GraphSAGE | tx-queue (no groups) | 0.05 | 472 | 0.824 | 0.524 | 0.184 | 79.91 | 0.19 | 284 |
| GraphSAGE | wallet-queue | 0.05 | 372 | 0.439 | 0.413 | 0.273 | 20.992 | 0.571 | 123 |
| GraphSAGE | flow(ours) | ours | 0.05 | 494 | 0.798 | 0.548 | 0.199 | 63.651 | 0.25 | 253 |
| GraphSAGE | conncomp | sum | 0.05 | 497 | 0.802 | 0.552 | 0.2 | 63.651 | 0.251 | 256 |
| GraphSAGE | ego1hop | mean | 0.05 | 370 | 0.55 | 0.411 | 0.217 | 24.598 | 0.484 | 54 |
| GraphSAGE | flow(ours) | max (ablation: plain score) | 0.05 | 495 | 0.82 | 0.549 | 0.194 | 68.513 | 0.233 | 262 |
| GraphSAGE | flow(ours) | mean (ablation: plain score) | 0.05 | 481 | 0.837 | 0.534 | 0.185 | 72.923 | 0.212 | 257 |
| GraphSAGE | flow(ours) | top3 (ablation: plain score) | 0.05 | 485 | 0.83 | 0.538 | 0.188 | 71.571 | 0.218 | 255 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.05 | 497 | 0.802 | 0.552 | 0.2 | 63.651 | 0.251 | 256 |
| GraphSAGE | flow(ours) | no hub filter (ablation) | 0.05 | 494 | 0.798 | 0.548 | 0.199 | 63.651 | 0.25 | 253 |
| GraphSAGE | flow(ours) | no size cap (ablation) | 0.05 | 494 | 0.798 | 0.548 | 0.199 | 63.651 | 0.25 | 253 |

## E4 Group-level proxy metrics (top-10 groups per test step)

| risk | method | n_groups | mean_size | median_size | max_size | hit_lenient | hit_strict | purity_known | label_coverage | null_purity | null_hit_lenient | enrichment |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LightGBM | flow(ours) | ours | 140 | 4.25 | 3.0 | 21.0 | 0.386 | 0.329 | 0.961 | 0.432 | 0.055 | 0.041 | 17.329 |
| LightGBM | conncomp | sum | 140 | 4.829 | 3.0 | 49.0 | 0.379 | 0.321 | 0.96 | 0.442 | 0.048 | 0.044 | 19.832 |
| LightGBM | ego1hop | mean | 140 | 2.986 | 3.0 | 6.0 | 0.429 | 0.314 | 0.899 | 0.38 | 0.047 | 0.034 | 18.958 |
| GraphSAGE | flow(ours) | ours | 140 | 3.357 | 2.0 | 36.0 | 0.293 | 0.214 | 0.92 | 0.372 | 0.051 | 0.04 | 18.126 |
| GraphSAGE | conncomp | sum | 140 | 3.586 | 2.0 | 36.0 | 0.314 | 0.221 | 0.926 | 0.376 | 0.057 | 0.037 | 16.111 |
| GraphSAGE | ego1hop | mean | 140 | 2.879 | 2.0 | 14.0 | 0.386 | 0.271 | 0.834 | 0.375 | 0.049 | 0.037 | 17.166 |

## E5 Counterfactual evidence deletion (GraphSAGE)

| m | groups | d_top | d_random | d_bottom | d_heuristic | sufficiency | top_beats_random_frac |
|---|---|---|---|---|---|---|---|
| 1.0 | 117.0 | 0.2559 | 0.0688 | 0.0294 | 0.3048 | 0.8687 | 0.5897 |
| 3.0 | 106.0 | 0.5999 | 0.2755 | 0.1699 | 0.4294 | 0.9363 | 0.6226 |
| 5.0 | 84.0 | 0.7795 | 0.4041 | 0.3124 | 0.4451 | 0.9543 | 0.619 |

## E6 Bootstrap 95% CI of differences (ours minus other)

| risk | budget | vs | illicit_diff | illicit_CI95 | cases_diff | cases_CI95 |
|---|---|---|---|---|---|---|
| LightGBM | 0.005 | tx-queue (no groups) | -7 | (-24.025, 10.0) | -141 | (-192.025, -98.97500000000014) |
| LightGBM | 0.005 | wallet-queue | 56 | (18.0, 98.0) | 11 | (-36.0, 63.0) |
| LightGBM | 0.005 | conncomp | sum | 1 | (-3.0, 6.0) | -19 | (-76.0, 16.0) |
| LightGBM | 0.005 | ego1hop | mean | 13 | (-7.024999999999999, 41.0) | 26 | (-10.024999999999999, 72.0) |
| LightGBM | 0.01 | tx-queue (no groups) | 2 | (-19.0, 26.0) | -296 | (-400.025, -212.0) |
| LightGBM | 0.01 | wallet-queue | 114 | (46.0, 188.0) | 34 | (-63.025, 133.04999999999973) |
| LightGBM | 0.01 | conncomp | sum | 5 | (-6.0, 16.0) | 25 | (10.0, 44.0) |
| LightGBM | 0.01 | ego1hop | mean | 43 | (16.0, 76.0) | 61 | (0.0, 126.0) |
| LightGBM | 0.02 | tx-queue (no groups) | 12 | (-23.0, 46.0) | -595 | (-751.025, -469.0) |
| LightGBM | 0.02 | wallet-queue | 167 | (53.975, 307.02499999999986) | 125 | (-23.0, 257.02499999999986) |
| LightGBM | 0.02 | conncomp | sum | 4 | (-14.0, 25.0) | 116 | (67.0, 172.0) |
| LightGBM | 0.02 | ego1hop | mean | 70 | (19.0, 130.0) | 152 | (58.0, 252.0) |
| LightGBM | 0.05 | tx-queue (no groups) | 20 | (3.0, 41.0) | -769 | (-972.0, -627.0) |
| LightGBM | 0.05 | wallet-queue | 175 | (57.975, 311.0) | 1050 | (816.975, 1261.0249999999999) |
| LightGBM | 0.05 | conncomp | sum | -1 | (-3.0, 0.0) | 99 | (50.0, 158.0) |
| LightGBM | 0.05 | ego1hop | mean | 126 | (26.975, 253.04999999999973) | 1098 | (896.975, 1293.0499999999997) |
| GraphSAGE | 0.005 | tx-queue (no groups) | 29 | (2.0, 60.0) | -154 | (-215.0, -99.0) |
| GraphSAGE | 0.005 | wallet-queue | 51 | (-5.024999999999999, 106.02499999999986) | 39 | (-12.0, 94.0) |
| GraphSAGE | 0.005 | conncomp | sum | 8 | (-1.0, 22.0) | 13 | (4.0, 23.0) |
| GraphSAGE | 0.005 | ego1hop | mean | 16 | (5.0, 29.0) | 49 | (15.975000000000001, 89.0) |
| GraphSAGE | 0.01 | tx-queue (no groups) | 28 | (-7.0, 66.0) | -282 | (-377.0, -202.0) |
| GraphSAGE | 0.01 | wallet-queue | 79 | (13.975000000000001, 151.0) | 93 | (28.975, 152.0) |
| GraphSAGE | 0.01 | conncomp | sum | -7 | (-16.0, 2.0) | 24 | (11.0, 40.0) |
| GraphSAGE | 0.01 | ego1hop | mean | 16 | (5.0, 29.0) | 108 | (58.975, 159.0) |
| GraphSAGE | 0.02 | tx-queue (no groups) | 44 | (3.0, 90.02499999999986) | -389 | (-520.025, -282.0) |
| GraphSAGE | 0.02 | wallet-queue | 139 | (25.975, 279.02499999999986) | 375 | (272.0, 473.02499999999986) |
| GraphSAGE | 0.02 | conncomp | sum | -2 | (-5.0, 0.0) | 16 | (7.0, 26.0) |
| GraphSAGE | 0.02 | ego1hop | mean | 58 | (7.0, 117.0) | 382 | (310.0, 449.0) |
| GraphSAGE | 0.05 | tx-queue (no groups) | 22 | (1.0, 44.0) | -505 | (-614.025, -394.0) |
| GraphSAGE | 0.05 | wallet-queue | 122 | (29.975, 235.0) | 1325 | (1110.95, 1518.0249999999999) |
| GraphSAGE | 0.05 | conncomp | sum | -3 | (-8.0, 0.0) | 0 | (0.0, 0.0) |
| GraphSAGE | 0.05 | ego1hop | mean | 124 | (35.0, 228.0) | 1213 | (1027.975, 1418.0) |
