"""Point-in-time (as-of) address features and the lifetime-leakage audit.

For an address ``a`` active at step ``t`` the features use

* the address's activity *in* step ``t`` (``cur_*``), observable at the end of the step, and
* its activity in *strictly earlier* steps (``prior_*``), and
* label history restricted to transactions that are (i) in the training period and (ii) in a
  strictly earlier step. Labels of the current step, of validation/test steps and of the
  transaction being scored are never used.

The official Elliptic++ wallet features are lifetime summaries repeated at every step, so
they contain future information; :func:`lifetime_leakage_audit` quantifies this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AS_OF_FEATURES = ["cur_send", "cur_recv", "cur_btc", "cur_fee", "cur_ntx", "prior_cur_send", "prior_cur_recv",
                  "prior_cur_ntx", "prior_steps", "age", "prior_ill", "prior_lic", "prior_ill_frac",
                  "prior_any_ill", "prior_mean_btc"]


def address_step_table(links: pd.DataFrame, train_end: int) -> pd.DataFrame:
    """Build the (address, step) table of as-of features.

    ``links`` needs columns ``a`` (address id), ``tx``, ``ts`` (step), ``role`` (0 = input/sender,
    1 = output/receiver), ``ylab`` (1 illicit, 0 licit, -1 unknown), ``btc`` and ``fee``.
    """
    L = links.copy()
    L["is_send"] = (L.role == 0).astype(np.int32)
    L["is_recv"] = (L.role == 1).astype(np.int32)
    L["ill_tr"] = ((L.ylab == 1) & (L.ts <= train_end)).astype(np.int32)
    L["lic_tr"] = ((L.ylab == 0) & (L.ts <= train_end)).astype(np.int32)
    A = (L.groupby(["a", "ts"])
         .agg(cur_send=("is_send", "sum"), cur_recv=("is_recv", "sum"), cur_btc=("btc", "sum"),
              cur_fee=("fee", "sum"), cur_ntx=("tx", "nunique"), ill=("ill_tr", "sum"), lic=("lic_tr", "sum"))
         .reset_index().sort_values(["a", "ts"]).reset_index(drop=True))
    g = A.groupby("a")
    for c in ["cur_send", "cur_recv", "cur_btc", "cur_ntx", "ill", "lic"]:
        A["prior_" + c] = g[c].cumsum() - A[c]          # strictly earlier steps only
    A["prior_steps"] = g.cumcount()
    A["age"] = A.ts - g.ts.transform("min")
    A["prior_ill_frac"] = A.prior_ill / (A.prior_ill + A.prior_lic).replace(0, np.nan)
    A["prior_any_ill"] = (A.prior_ill > 0).astype(np.float32)
    A["prior_mean_btc"] = A.prior_cur_btc / A.prior_cur_ntx.replace(0, np.nan)
    return A


def aggregate_to_transactions(links: pd.DataFrame, table: pd.DataFrame, n_tx: int, cols=None, prefix="xv"):
    """Mean and max of address features over each transaction's input and output addresses."""
    cols = cols or AS_OF_FEATURES
    LA = links[["tx", "a", "ts", "role"]].merge(table[["a", "ts"] + cols], on=["a", "ts"], how="left")
    out = []
    for r, name in [(0, "in"), (1, "out")]:
        d = LA[LA.role == r].groupby("tx")[cols].agg(["mean", "max"])
        d.columns = [f"{prefix}_{name}_{c}_{s}" for c, s in d.columns]
        out.append(d)
    return pd.concat(out, axis=1).reindex(np.arange(n_tx)).astype(np.float32)


def lifetime_leakage_audit(wallet: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Quantify future information in per-step wallet rows.

    ``wallet`` has columns ``a, ts, total_txs, last_block_appeared_in, num_timesteps_appeared_in``;
    ``table`` is the output of :func:`address_step_table`.
    """
    cnt = wallet.groupby("a").ts.transform("size")
    multi = wallet[cnt > 1]
    const = multi.groupby("a")[["total_txs", "last_block_appeared_in", "num_timesteps_appeared_in"]].nunique()
    chk = wallet.merge(table[["a", "ts", "prior_steps"]], on=["a", "ts"], how="inner")
    return dict(
        rows=int(len(wallet)),
        addresses_multi_step=int(const.shape[0]),
        frac_multi_step_addr_constant_total_txs=float((const.total_txs == 1).mean()) if len(const) else float("nan"),
        frac_multi_step_addr_constant_last_block=float((const.last_block_appeared_in == 1).mean()) if len(const) else float("nan"),
        frac_rows_counting_future_steps=float((chk.num_timesteps_appeared_in > chk.prior_steps + 1).mean()) if len(chk) else float("nan"),
    )
