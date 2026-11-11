"""Re-draw every result chart from the CSV tables in results/ (no data download needed).

    python scripts/make_figures.py

Writes PNGs to figures/. Colours use a CVD-validated categorical palette; every multi-series
chart carries a legend plus direct labels and distinct marker shapes (secondary encoding).
"""
from __future__ import annotations

import ast
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#bdbcb7", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK, "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 7,
})

QUEUES = [  # (name in CSV, label, colour, marker)
    ("flow(ours) | ours", "FLARE groups (ours)", BLUE, "o"),
    ("tx-queue (no groups)", "Transaction queue", ORANGE, "s"),
    ("conncomp | sum", "Connected components", AQUA, "^"),
    ("ego1hop | mean", "1-hop neighbourhood", YELLOW, "D"),
    ("wallet-queue", "Wallet queue", MAGENTA, "v"),
]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def read(name):
    return pd.read_csv(os.path.join(RES, name))


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center", fontsize=8, color=INK2)


# ------------------------------------------------------------------------------------------- data
def fig_labels():
    d = read("label_counts_per_step.csv")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.bar(d.ts, d.unknown, color="#d4d3ce", width=0.8, label="unknown", edgecolor="white", linewidth=0.6)
    ax.bar(d.ts, d.licit, bottom=d.unknown, color=AQUA, width=0.8, label="licit", edgecolor="white", linewidth=0.6)
    ax.bar(d.ts, d.illicit, bottom=d.unknown + d.licit, color=RED, width=0.8, label="illicit", edgecolor="white", linewidth=0.6)
    for x0, x1, lab in [(0.5, 30.5, "train"), (30.5, 35.5, "validation"), (35.5, 49.5, "test")]:
        ax.axvspan(x0, x1, color={"train": "#eef4fc", "validation": "#fff4dd", "test": "#e9f7f0"}[lab], zorder=0, alpha=0.8)
        ax.text((x0 + x1) / 2 - (2.5 if lab == "test" else 0), d.total.max() * 1.05, lab, ha="center", fontsize=9, color=INK2)
    ax.axvline(42.5, color=RED, ls="--", lw=1.2)
    ax.text(42.7, d.total.max() * 0.97, "shutdown", fontsize=8, color=RED)
    ax.set_xlabel("time step"); ax.set_ylabel("transactions"); ax.set_xlim(0.3, 49.7)
    ax.set_ylim(0, d.total.max() * 1.12); ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0), fontsize=8.5)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_label_distribution.png")


def fig_illicit_rate():
    d = read("label_counts_per_step.csv")
    rate = d.illicit / (d.illicit + d.licit)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(d.ts, 100 * rate, color=RED, marker="o", ms=4)
    ax.axvline(42.5, color=RED, ls="--", lw=1.0)
    ax.set_xlabel("time step"); ax.set_ylabel("illicit share of\nlabelled tx (%)"); ax.set_xlim(0.3, 49.7)
    save(fig, "fig_illicit_rate.png")


# ----------------------------------------------------------------------------------------- models
def fig_models():
    d = read("E1_tx_models_summary.csv")
    d["name"] = d.model.str.replace("SAGE-dir", "GraphSAGE (dir.)").str.replace("SAGE-undir", "GraphSAGE (undir.)") + "  ·  " + \
        d.features.str.replace("XVlifetime", "XV-lifetime")
    d = d.sort_values("AP_mean")
    col = [BLUE if m == "LightGBM" else (VIOLET if "SAGE" in m else (AQUA if m == "RF" else MUTED)) for m in d.model]
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.barh(d.name, d.AP_mean, xerr=d.AP_std.fillna(0), color=col, height=0.62, error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    for y, (v, e) in enumerate(zip(d.AP_mean, d.AP_std.fillna(0))):
        ax.text(v + e + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("test PR-AUC (mean ± sd over seeds)"); ax.set_xlim(0, 0.85); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    save(fig, "fig_model_comparison.png")


def fig_per_step():
    d = read("E6_per_step.csv")
    series = [("LightGBM", "AF+AUG+XV", "LightGBM (AF+AUG+XV)", BLUE, "o"), ("RF", "AF", "Random forest (AF)", AQUA, "^"),
              ("SAGE-dir", "AF+AUG+XV", "GraphSAGE (AF+AUG+XV)", VIOLET, "s"), ("MLP", "AF+AUG", "MLP (AF+AUG)", ORANGE, "D")]
    fig, ax = plt.subplots(figsize=(10, 3.9))
    for m, f, lab, c, mk in series:
        s = d[(d.model == m) & (d.features == f)].sort_values("step")
        ax.plot(s.step, s.F1, color=c, marker=mk, ms=5, label=lab)
    ax.axvline(42.5, color=RED, ls="--", lw=1.2)
    ax.text(42.7, 0.95, "dark-market\nshutdown", fontsize=8, color=RED, va="top")
    ax.set_xlabel("test time step"); ax.set_ylabel("illicit F1"); ax.set_ylim(0, 1.02)
    ax.legend(ncol=1, fontsize=8.5, loc="upper right")
    save(fig, "fig_per_step_f1.png")


def fig_views_and_wallet():
    e1 = read("E1_tx_models_summary.csv"); e2 = read("E2_wallet_models_summary.csv")
    v = e1[e1.model == "LightGBM"].set_index("features").loc[["LF", "AF", "AF+AUG", "AF+AUG+XV", "AF+AUG+XVlifetime"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    cols = [MUTED, MUTED, MUTED, BLUE, RED]
    ax[0].bar(range(5), v.AP_mean, yerr=v.AP_std, color=cols, width=0.62, error_kw=dict(ecolor=INK2, capsize=3))
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(["LF", "AF", "AF+AUG", "+XV\n(as-of)", "+XV\n(lifetime)"], fontsize=8.5)
    ax[0].set_ylim(0.70, 0.76); ax[0].set_ylabel("test PR-AUC"); ax[0].set_title("(a) Transactions: LightGBM by feature view", loc="left", fontsize=10)
    for i, (val, e) in enumerate(zip(v.AP_mean, v.AP_std)):
        ax[0].text(i, val + e + 0.0008, f"{val:.3f}", ha="center", fontsize=8, color=INK2)
    w = e2.set_index("features").loc[["W-asof", "W-official(lifetime)", "W-asof+txrisk", "W-official+txrisk(lifetime)"]]
    ax[1].bar(range(4), w.addr_AP_mean, yerr=w.addr_AP_std, color=[BLUE, RED, BLUE, RED], width=0.62,
              error_kw=dict(ecolor=INK2, capsize=3))
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["as-of", "lifetime", "as-of\n+ tx risk", "lifetime\n+ tx risk"], fontsize=8.5)
    ax[1].set_ylabel("address-level test PR-AUC"); ax[1].set_ylim(0, 0.75)
    ax[1].set_title("(b) Wallets: as-of vs official lifetime features", loc="left", fontsize=10)
    for i, (val, e) in enumerate(zip(w.addr_AP_mean, w.addr_AP_std)):
        ax[1].text(i, val + e + 0.012, f"{val:.3f}", ha="center", fontsize=8, color=INK2)
    for a in ax:
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_feature_views_and_wallet.png")


# ----------------------------------------------------------------------------------------- queues
def fig_queues(risk):
    d = read("E3_queues.csv"); d = d[d.risk == risk]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.1))
    metrics = [("illicit_found", "illicit transactions found"), ("cases_per_100", "cases per 100 reviewed (lower = less fragmented)"),
               ("illicit_per_case", "illicit transactions per case")]
    for k, (m, lab) in enumerate(metrics):
        for q, name, c, mk in QUEUES:
            s = d[d.queue == q].sort_values("budget")
            ax[k].plot(s.budget * 100, s[m], color=c, marker=mk, lw=2.6 if "ours" in q else 1.6, label=name)
        ax[k].set_xlabel("review budget (% of transactions per step)"); ax[k].set_title(lab, loc="left", fontsize=9.5)
        ax[k].set_xticks([0.5, 1, 2, 5])
    ax[0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Review queues on test steps 36-49 ({risk} risk scores)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, f"fig_queues_{risk}.png")


def fig_bootstrap():
    d = read("E6_bootstrap_queue_diffs.csv")
    d = d[d.vs == "tx-queue (no groups)"].copy()
    for c in ["illicit_CI95", "cases_CI95"]:
        d[c] = d[c].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for k, (val, ci, lab) in enumerate([("illicit_diff", "illicit_CI95", "illicit found: groups − transaction queue"),
                                        ("cases_diff", "cases_CI95", "cases: groups − transaction queue")]):
        ylabels = []
        for i, (_, r) in enumerate(d.sort_values(["risk", "budget"]).iterrows()):
            c = BLUE if r.risk == "LightGBM" else VIOLET
            mk = "o" if r.risk == "LightGBM" else "s"
            ax[k].plot(r[ci], [i, i], color=c, lw=2)
            ax[k].plot(r[val], i, marker=mk, color=c, ms=7)
            ylabels.append(f"{r.risk} @ {r.budget * 100:g}%")
        ax[k].axvline(0, color=INK2, lw=1)
        ax[k].set_yticks(range(len(ylabels))); ax[k].set_yticklabels(ylabels, fontsize=8.5)
        ax[k].set_title(lab, loc="left", fontsize=9.5); ax[k].grid(axis="y", visible=False)
        ax[k].set_xlabel("difference with 95% paired-bootstrap CI (14 test steps)")
    fig.tight_layout()
    save(fig, "fig_bootstrap_ci.png")


def fig_scarcity():
    d = read("X2_label_scarcity_summary.csv").sort_values("label_frac")
    fr = d.label_frac * 100
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.9))
    ax[0].errorbar(fr, d.node_AP_mean, d.node_AP_std.fillna(0), color=INK2, marker="o", capsize=3)
    ax[0].set_title("transaction PR-AUC", loc="left", fontsize=9.5)
    for tag, name, c, mk in [("flow", "FLARE groups (ours)", BLUE, "o"), ("tx", "Transaction queue", ORANGE, "s"),
                             ("cc", "Connected components", AQUA, "^")]:
        ax[1].errorbar(fr, d[f"{tag}_ill@0.01_mean"], d[f"{tag}_ill@0.01_std"].fillna(0), color=c, marker=mk, capsize=3,
                       label=name, lw=2.6 if tag == "flow" else 1.6)
        ax[2].errorbar(fr, d[f"{tag}_cases100@0.01_mean"], d[f"{tag}_cases100@0.01_std"].fillna(0), color=c, marker=mk,
                       capsize=3, label=name, lw=2.6 if tag == "flow" else 1.6)
    ax[1].set_title("illicit found at 1% budget", loc="left", fontsize=9.5)
    ax[2].set_title("cases per 100 reviewed at 1% budget", loc="left", fontsize=9.5)
    ax[1].legend(fontsize=8, loc="lower right")
    for a in ax:
        a.set_xscale("log"); a.set_xticks(list(fr)); a.set_xticklabels([f"{int(f)}%" for f in fr]); a.minorticks_off()
        a.set_xlabel("training labels used")
    fig.tight_layout()
    save(fig, "fig_label_scarcity.png")


def fig_enrichment():
    d = read("E4_group_proxy_metrics.csv")
    d["lab"] = d.risk + "\n" + d.method.str.replace("flow(ours) | ours", "FLARE (ours)", regex=False) \
        .str.replace("conncomp | sum", "conn. comp.", regex=False).str.replace("ego1hop | mean", "1-hop", regex=False)
    x = np.arange(len(d)); wdt = 0.38
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(x - wdt / 2, d.purity_known, wdt, color=BLUE, label="top-10 groups per step")
    ax.bar(x + wdt / 2, d.null_purity, wdt, color="#bdbcb7", label="size-matched random groups")
    for i, r in d.reset_index().iterrows():
        ax.text(i - wdt / 2, r.purity_known + 0.02, f"{r.enrichment:.0f}x", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(d.lab, fontsize=8.5); ax.set_ylabel("illicit share of labelled members")
    ax.set_ylim(0, 1.08); ax.legend(fontsize=8.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2); ax.grid(axis="x", visible=False)
    save(fig, "fig_enrichment.png")


def fig_counterfactual():
    d = read("E5_counterfactual_summary.csv")
    x = np.arange(len(d)); wdt = 0.2
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (c, lab, col) in enumerate([("d_top", "gradient-top edges", VIOLET), ("d_heuristic", "top-risk-member heuristic", ORANGE),
                                       ("d_random", "random edges", "#bdbcb7"), ("d_bottom", "gradient-bottom edges", AQUA)]):
        ax.bar(x + (i - 1.5) * wdt, d[c], wdt, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"m = {int(m)} edges deleted" for m in d.m])
    ax.set_ylabel("drop in group score"); ax.legend(fontsize=8.5, loc="upper left"); ax.grid(axis="x", visible=False)
    save(fig, "fig_counterfactual.png")


if __name__ == "__main__":
    fig_labels(); fig_illicit_rate(); fig_models(); fig_per_step(); fig_views_and_wallet()
    fig_queues("LightGBM"); fig_queues("GraphSAGE"); fig_bootstrap(); fig_scarcity(); fig_enrichment(); fig_counterfactual()
