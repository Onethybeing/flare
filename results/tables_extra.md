# Supplementary tables

## X1 Extended tuning grid (sfm up to 3.0)

| risk | queue | budget | illicit_found | precision_known | recall | label_coverage | cases_per_100 | illicit_per_case | illicit_cases |
|---|---|---|---|---|---|---|---|---|---|
| LightGBM | tx-queue (no groups) | 0.005 | 115 | 0.983 | 0.128 | 0.371 | 81.905 | 0.446 | 85 |
| LightGBM | flow(ours) | ours (sfm=2.0) | 0.005 | 107 | 0.982 | 0.119 | 0.346 | 58.095 | 0.585 | 42 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.005 | 110 | 0.973 | 0.122 | 0.359 | 39.365 | 0.887 | 26 |
| LightGBM | conncomp | sum (sfm=2.0) | 0.005 | 109 | 0.956 | 0.121 | 0.362 | 37.46 | 0.924 | 24 |
| LightGBM | tx-queue (no groups) | 0.01 | 208 | 0.986 | 0.231 | 0.337 | 83.706 | 0.397 | 152 |
| LightGBM | flow(ours) | ours (sfm=2.0) | 0.01 | 209 | 0.981 | 0.232 | 0.34 | 64.856 | 0.515 | 103 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.01 | 205 | 0.958 | 0.228 | 0.342 | 41.214 | 0.795 | 64 |
| LightGBM | conncomp | sum (sfm=2.0) | 0.01 | 203 | 0.958 | 0.225 | 0.339 | 40.096 | 0.809 | 60 |
| LightGBM | tx-queue (no groups) | 0.02 | 326 | 0.985 | 0.362 | 0.265 | 84.683 | 0.309 | 244 |
| LightGBM | flow(ours) | ours (sfm=2.0) | 0.02 | 345 | 0.983 | 0.383 | 0.281 | 73.536 | 0.376 | 209 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.02 | 342 | 0.969 | 0.38 | 0.283 | 50.922 | 0.539 | 160 |
| LightGBM | conncomp | sum (sfm=2.0) | 0.02 | 343 | 0.969 | 0.381 | 0.284 | 50.521 | 0.544 | 161 |
| LightGBM | tx-queue (no groups) | 0.05 | 527 | 0.923 | 0.585 | 0.184 | 79.588 | 0.213 | 342 |
| LightGBM | flow(ours) | ours (sfm=2.0) | 0.05 | 516 | 0.894 | 0.573 | 0.186 | 65.937 | 0.252 | 253 |
| LightGBM | flow(ours) | sum (ablation: plain score) | 0.05 | 535 | 0.89 | 0.594 | 0.193 | 50.901 | 0.338 | 238 |
| LightGBM | conncomp | sum (sfm=2.0) | 0.05 | 535 | 0.89 | 0.594 | 0.193 | 50.901 | 0.338 | 238 |
| GraphSAGE | tx-queue (no groups) | 0.005 | 74 | 0.961 | 0.082 | 0.244 | 95.238 | 0.247 | 69 |
| GraphSAGE | flow(ours) | ours (sfm=1.5) | 0.005 | 94 | 0.94 | 0.104 | 0.317 | 48.571 | 0.614 | 27 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.005 | 96 | 0.941 | 0.107 | 0.324 | 48.571 | 0.627 | 28 |
| GraphSAGE | conncomp | sum (sfm=1.5) | 0.005 | 92 | 0.902 | 0.102 | 0.324 | 40.635 | 0.719 | 19 |
| GraphSAGE | tx-queue (no groups) | 0.01 | 147 | 0.955 | 0.163 | 0.246 | 89.936 | 0.261 | 121 |
| GraphSAGE | flow(ours) | ours (sfm=1.5) | 0.01 | 171 | 0.914 | 0.19 | 0.299 | 50.639 | 0.539 | 67 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.01 | 172 | 0.915 | 0.191 | 0.3 | 50.639 | 0.543 | 68 |
| GraphSAGE | conncomp | sum (sfm=1.5) | 0.01 | 185 | 0.907 | 0.205 | 0.326 | 39.776 | 0.743 | 53 |
| GraphSAGE | tx-queue (no groups) | 0.02 | 260 | 0.949 | 0.289 | 0.22 | 88.532 | 0.236 | 207 |
| GraphSAGE | flow(ours) | ours (sfm=1.5) | 0.02 | 298 | 0.9 | 0.331 | 0.265 | 58.861 | 0.406 | 137 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.02 | 299 | 0.903 | 0.332 | 0.265 | 58.861 | 0.407 | 138 |
| GraphSAGE | conncomp | sum (sfm=1.5) | 0.02 | 304 | 0.897 | 0.337 | 0.272 | 55.253 | 0.441 | 128 |
| GraphSAGE | tx-queue (no groups) | 0.05 | 467 | 0.828 | 0.518 | 0.182 | 79.62 | 0.189 | 283 |
| GraphSAGE | flow(ours) | ours (sfm=1.5) | 0.05 | 496 | 0.81 | 0.55 | 0.197 | 63.941 | 0.25 | 254 |
| GraphSAGE | flow(ours) | sum (ablation: plain score) | 0.05 | 497 | 0.813 | 0.552 | 0.197 | 63.941 | 0.25 | 255 |
| GraphSAGE | conncomp | sum (sfm=1.5) | 0.05 | 497 | 0.813 | 0.552 | 0.197 | 63.941 | 0.25 | 255 |

## X2 Label scarcity (mean over 3 random label subsets)

| label_frac | n_train_labels_mean | n_train_labels_std | n_train_illicit_mean | n_train_illicit_std | node_AP_mean | node_AP_std | tx_ill@0.01_mean | tx_ill@0.01_std | tx_cases100@0.01_mean | tx_cases100@0.01_std | flow_ill@0.01_mean | flow_ill@0.01_std | flow_cases100@0.01_mean | flow_cases100@0.01_std | cc_ill@0.01_mean | cc_ill@0.01_std | cc_cases100@0.01_mean | cc_cases100@0.01_std | tx_ill@0.02_mean | tx_ill@0.02_std | tx_cases100@0.02_mean | tx_cases100@0.02_std | flow_ill@0.02_mean | flow_ill@0.02_std | flow_cases100@0.02_mean | flow_cases100@0.02_std | cc_ill@0.02_mean | cc_ill@0.02_std | cc_cases100@0.02_mean | cc_cases100@0.02_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 2690.0 | 0.0 | 295.0 | 0.0 | 0.722 | 0.003 | 175.333 | 10.97 | 87.061 | 2.826 | 201.333 | 11.547 | 32.961 | 4.953 | 203.333 | 10.97 | 30.085 | 4.592 | 292.0 | 22.113 | 86.902 | 2.408 | 320.333 | 16.073 | 35.686 | 6.169 | 322.667 | 17.039 | 32.13 | 5.992 |
| 0.2 | 5381.0 | 0.0 | 591.0 | 0.0 | 0.732 | 0.008 | 170.333 | 9.018 | 86.954 | 3.658 | 199.667 | 5.859 | 36.528 | 5.516 | 203.333 | 3.055 | 33.12 | 4.243 | 275.667 | 4.163 | 87.891 | 0.212 | 316.667 | 9.074 | 39.428 | 6.744 | 321.333 | 14.64 | 31.222 | 5.94 |
| 0.5 | 13452.0 | 0.0 | 1477.0 | 0.0 | 0.753 | 0.005 | 193.333 | 14.012 | 84.292 | 0.605 | 212.0 | 5.0 | 34.931 | 0.788 | 210.667 | 4.163 | 30.298 | 0.908 | 321.0 | 7.211 | 86.18 | 0.668 | 340.0 | 7.211 | 38.893 | 1.995 | 342.333 | 8.622 | 29.377 | 1.364 |
| 1.0 | 26905.0 | 0.0 | 2954.0 | 0.0 | 0.75 | 0.0 | 207.667 | 3.055 | 83.28 | 0.514 | 207.333 | 5.033 | 35.25 | 0.244 | 203.333 | 8.083 | 30.884 | 0.665 | 329.0 | 11.79 | 84.71 | 0.546 | 341.333 | 3.215 | 36.995 | 1.146 | 338.333 | 2.082 | 27.854 | 0.762 |
