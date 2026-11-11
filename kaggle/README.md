# Kaggle notebook sources

These are the exact sources that produced every number in `results/`.

| Folder | Notebook | Purpose |
|---|---|---|
| `probe/` | `make_probe.py` | prints the schema of all Elliptic++ files (first run) |
| `main/` | `btp_main.py` + `make_main.py [quick|full]` | E1-E6: models, wallet study, leakage audit, queues, group metrics, counterfactuals, bootstrap |
| `extra/` | `extra_body.py` + `make_extra.py [quick|full]` | X1 wider tuning grid, X2 label scarcity |
| `polish/` | `make_polish.py` | risk scores + readable case-study figures |
| `nbtools.py` | | writes `.ipynb` and `kernel-metadata.json` (dataset `muhammadabrar78/elliptic-dataset`) |
| `fetch_run.sh` | | downloads a finished kernel's `results/`, `figures/` and log into `runs/<slug>_<time>/` |

`btp_main.py` is organised in numbered cells (`# %% [0]` ... `# %% [13]`):
0 setup, 1 transactions, 2 as-of address features, 3 official wallet features and leakage audit, 4 feature sets,
5 E1 tabular models, 6 E1 GraphSAGE, 7 E2 wallets, 8 step graph, 9 group builders and queues, 10 E3 tuning and queues,
11 E4 proxy metrics, 12 E5 counterfactuals, 13 E6 bootstrap, figures and tables.

The tested, documented versions of the core algorithms live in `src/flowgroups/`; the notebook keeps its own copies so
that the executed code stays exactly as run.
