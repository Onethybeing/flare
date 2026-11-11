# Reproducibility

## What produced each artefact

| Artefact | Produced by | Where it ran |
|---|---|---|
| `results/tables.md`, `results/E1_*` to `E6_*`, `summary.json` | `kaggle/main/btp_main.py` (notebook built by `make_main.py full`) | Kaggle, T4 GPU, 30.9 min |
| `results/tables_extra.md`, `X1_*`, `X2_*`, `summary_extra.json` | `kaggle/main/btp_main.py` sections + `kaggle/extra/extra_body.py` (`make_extra.py full`) | Kaggle, T4 GPU, 11.2 min |
| `figures/fig_case_study_group_*.png`, `results/case_study_group_*.csv` | `kaggle/polish/make_polish.py` | Kaggle, CPU, 2.5 min |
| `results/label_counts_per_step.csv` | aggregated from the saved labels (counts only) | local |
| `figures/*.png` (all other charts) | `scripts/make_figures.py` from the CSVs | local, seconds |
| `docs/assets/*.png` | `scripts/make_diagrams.py` | local, seconds |

The notebooks are generated from plain `.py` sources: `make_*.py` splits a script on `# %% [k]` markers into
notebook cells and writes `kernel-metadata.json`. Edit the `.py`, rebuild, push. Set `"id"` in
`kaggle/nbtools.py:write_meta` (currently `onethybeing/...`) to your own Kaggle username.

## Environment

* Kaggle image (November 2026): Python 3.13, LightGBM 4.6.0, NetworkX 3.7, PyTorch 2.11 (CUDA 13), scikit-learn, pandas, SciPy.
* Package and tests: Python 3.10+, numpy, scipy, pandas, networkx; PyTorch optional (model and counterfactual tests are skipped without it). scikit-learn is needed only for `best_f1_threshold`.
* Local verification: Windows 11, Python 3.11, PyTorch 2.9 CPU: `python -m unittest discover -s tests` runs 35 tests.

## Determinism

* LightGBM seeds 0-4 (`random_state`), random forest seeds 0-2, GraphSAGE `torch.manual_seed(0..4)`.
* Label subsets in X2 use `numpy.random.default_rng(100 + rep)`; nulls in E4 use `default_rng(0)`; bootstrap uses `default_rng(1)`.
* GPU `index_add_` is not bit-deterministic, so GraphSAGE scores can differ in the last digits between runs; LightGBM risk
  reproduced exactly (test PR-AUC 0.7505 in both the main and the figure run).

## Steps

```bash
pip install kaggle                      # and place your API token in ~/.kaggle/
cd kaggle/probe && python make_probe.py && kaggle kernels push -p .      # optional schema probe
cd ../main && python make_main.py quick && kaggle kernels push -p .      # 7-minute smoke test (1 seed)
python make_main.py full && kaggle kernels push -p .                     # full run (5 seeds)
cd ../extra && python make_extra.py full && kaggle kernels push -p .     # X1 + X2
bash ../fetch_run.sh btp-flowgroup-full                                  # download results/ and figures/
```

## Runtime budget (T4)

| Stage | Minutes |
|---|---|
| Data loading + as-of features + leakage audit | 2 |
| E1 transaction models (5 feature sets x 5 seeds LightGBM, RF, LR, 4 GNN variants x 5 seeds) | 14 |
| E2 wallet models + out-of-fold transaction risk | 4 |
| E3-E6 tuning, queues, nulls, counterfactuals, bootstrap, figures | 11 |
