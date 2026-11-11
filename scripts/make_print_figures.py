"""Print-size figures for the report: drawn at the final text width (6.0 in) so that every label is at least 8 pt
when printed (IIT Kharagpur guideline). Reads only the CSV tables in results/.

    python scripts/make_print_figures.py          -> figures_print/*.png
"""
from __future__ import annotations

import ast
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, OUT = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures_print")
os.makedirs(OUT, exist_ok=True)
W = 6.0  # inches = report text width

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, LINE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#d9d8d3"
ILLICIT, LICIT, UNKNOWN = "#d64545", "#2f9e57", "#bdbdbd"
FS = 8.5
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "axes.edgecolor": "#bdbcb7", "axes.linewidth": 0.7, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5,
})
QUEUES = [("flow(ours) | ours", "FLARE groups (ours)", BLUE, "o"), ("tx-queue (no groups)", "Transaction queue", ORANGE, "s"),
          ("conncomp | sum", "Connected components", AQUA, "^"), ("ego1hop | mean", "1-hop neighbourhood", YELLOW, "D"),
          ("wallet-queue", "Wallet queue", MAGENTA, "v")]
read = lambda n: pd.read_csv(os.path.join(RES, n))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# =============================================================================================== charts
def labels():
    d = read("label_counts_per_step.csv")
    fig, ax = plt.subplots(figsize=(W, 2.5))
    for x0, x1, lab, c in [(0.5, 30.5, "train", "#eef4fc"), (30.5, 35.5, "val.", "#fff4dd"), (35.5, 49.5, "test", "#e9f7f0")]:
        ax.axvspan(x0, x1, color=c, zorder=0)
        ax.text((x0 + x1) / 2, d.total.max() * 1.03, lab, ha="center", fontsize=8, color=INK2)
    ax.bar(d.ts, d.unknown, color="#d4d3ce", width=0.8, label="unknown", edgecolor="white", lw=0.4)
    ax.bar(d.ts, d.licit, bottom=d.unknown, color=AQUA, width=0.8, label="licit", edgecolor="white", lw=0.4)
    ax.bar(d.ts, d.illicit, bottom=d.unknown + d.licit, color=RED, width=0.8, label="illicit", edgecolor="white", lw=0.4)
    ax.axvline(42.5, color=RED, ls="--", lw=1)
    ax.set_xlabel("time step"); ax.set_ylabel("transactions"); ax.set_xlim(0.3, 49.7); ax.set_ylim(0, d.total.max() * 1.12)
    ax.legend(ncol=3, loc="lower right", bbox_to_anchor=(1, 1.0)); ax.grid(axis="x", visible=False)
    save(fig, "fig_label_distribution.png")


def illicit_rate():
    d = read("label_counts_per_step.csv")
    fig, ax = plt.subplots(figsize=(W, 1.8))
    ax.plot(d.ts, 100 * d.illicit / (d.illicit + d.licit), color=RED, marker="o", ms=3)
    ax.axvline(42.5, color=RED, ls="--", lw=1)
    ax.set_xlabel("time step"); ax.set_ylabel("illicit share (%)"); ax.set_xlim(0.3, 49.7)
    save(fig, "fig_illicit_rate.png")


def models():
    d = read("E1_tx_models_summary.csv")
    rn = {"SAGE-dir": "GraphSAGE dir.", "SAGE-undir": "GraphSAGE undir.", "LogReg": "LogReg", "RF": "RF", "LightGBM": "LightGBM", "MLP": "MLP"}
    d["name"] = d.model.map(rn) + "  " + d.features.str.replace("XVlifetime", "XV-lifetime")
    d = d.sort_values("AP_mean")
    col = [BLUE if m == "LightGBM" else (VIOLET if "SAGE" in m else (AQUA if m == "RF" else MUTED)) for m in d.model]
    fig, ax = plt.subplots(figsize=(5.3, 3.9))
    ax.barh(d.name, d.AP_mean, xerr=d.AP_std.fillna(0), color=col, height=0.65, error_kw=dict(ecolor=INK2, lw=0.8, capsize=1.5))
    for y_, (v, e) in enumerate(zip(d.AP_mean, d.AP_std.fillna(0))):
        ax.text(v + e + 0.008, y_, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("test PR-AUC (mean ± sd over seeds)"); ax.set_xlim(0, 0.84); ax.grid(axis="y", visible=False)
    save(fig, "fig_model_comparison.png")


def per_step():
    d = read("E6_per_step.csv")
    fig, ax = plt.subplots(figsize=(W, 2.6))
    for m, f, lab, c, mk in [("LightGBM", "AF+AUG+XV", "LightGBM (AF+AUG+XV)", BLUE, "o"), ("RF", "AF", "Random forest (AF)", AQUA, "^"),
                             ("SAGE-dir", "AF+AUG+XV", "GraphSAGE (AF+AUG+XV)", VIOLET, "s"), ("MLP", "AF+AUG", "MLP (AF+AUG)", ORANGE, "D")]:
        s = d[(d.model == m) & (d.features == f)].sort_values("step")
        ax.plot(s.step, s.F1, color=c, marker=mk, label=lab)
    ax.axvline(42.5, color=RED, ls="--", lw=1); ax.text(42.35, 0.5, "shutdown", fontsize=8, color=RED, ha="right")
    ax.set_xlabel("test time step"); ax.set_ylabel("illicit F1"); ax.set_ylim(0, 1.02); ax.legend(loc="upper right")
    save(fig, "fig_per_step_f1.png")


def views():
    e1, e2 = read("E1_tx_models_summary.csv"), read("E2_wallet_models_summary.csv")
    v = e1[e1.model == "LightGBM"].set_index("features").loc[["LF", "AF", "AF+AUG", "AF+AUG+XV", "AF+AUG+XVlifetime"]]
    w = e2.set_index("features").loc[["W-asof", "W-official(lifetime)", "W-asof+txrisk", "W-official+txrisk(lifetime)"]]
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.6))
    ax[0].bar(range(5), v.AP_mean, yerr=v.AP_std, color=[MUTED] * 3 + [BLUE, RED], width=0.65, error_kw=dict(ecolor=INK2, capsize=2))
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(["LF", "AF", "AF+\nAUG", "+XV\nas-of", "+XV\nlifetime"])
    ax[0].set_ylim(0.70, 0.765); ax[0].set_ylabel("test PR-AUC"); ax[0].set_title("(a) transactions, LightGBM", loc="left")
    ax[1].bar(range(4), w.addr_AP_mean, yerr=w.addr_AP_std, color=[BLUE, RED, BLUE, RED], width=0.65, error_kw=dict(ecolor=INK2, capsize=2))
    ax[1].set_xticks(range(4)); ax[1].set_xticklabels(["as-of", "lifetime", "as-of\n+tx risk", "lifetime\n+tx risk"])
    ax[1].set_ylim(0, 0.75); ax[1].set_ylabel("address PR-AUC"); ax[1].set_title("(b) wallets", loc="left")
    for a, vals, errs, off in [(ax[0], v.AP_mean, v.AP_std, 0.001), (ax[1], w.addr_AP_mean, w.addr_AP_std, 0.015)]:
        for i, (val, e) in enumerate(zip(vals, errs)):
            a.text(i, val + e + off, f"{val:.3f}", ha="center", fontsize=8, color=INK2)
        a.grid(axis="x", visible=False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_feature_views_and_wallet.png")


def queues(risk):
    d = read("E3_queues.csv"); d = d[d.risk == risk]
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.6))
    panels = [("illicit_found", "(a) illicit transactions found"), ("cases_per_100", "(b) cases per 100 reviews"),
              ("illicit_per_case", "(c) illicit transactions per case")]
    for k, (m, lab) in enumerate(panels):
        ax = axs.flat[k]
        for q, name, c, mk in QUEUES:
            s = d[d.queue == q].sort_values("budget")
            ax.plot(s.budget * 100, s[m], color=c, marker=mk, lw=2.2 if "ours" in q else 1.3, label=name)
        ax.set_title(lab, loc="left"); ax.set_xticks([0.5, 1, 2, 5]); ax.set_xlabel("budget (% of step)")
    axs.flat[3].axis("off")
    h, l = axs.flat[0].get_legend_handles_labels()
    axs.flat[3].legend(h, l, loc="center", fontsize=8.5, title=f"{risk} risk scores", title_fontsize=8.5)
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, f"fig_queues_{risk}.png")


def bootstrap():
    d = read("E6_bootstrap_queue_diffs.csv"); d = d[d.vs == "tx-queue (no groups)"].sort_values(["risk", "budget"]).reset_index(drop=True)
    for c in ["illicit_CI95", "cases_CI95"]:
        d[c] = d[c].apply(ast.literal_eval)
    fig, ax = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
    ylab = [f"{r} {b * 100:g}%" for r, b in zip(d.risk, d.budget)]
    for k, (val, ci, lab) in enumerate([("illicit_diff", "illicit_CI95", "(a) illicit found"), ("cases_diff", "cases_CI95", "(b) number of cases")]):
        for i, r in d.iterrows():
            c, mk = (BLUE, "o") if r.risk == "LightGBM" else (VIOLET, "s")
            ax[k].plot(r[ci], [i, i], color=c, lw=1.6); ax[k].plot(r[val], i, marker=mk, color=c, ms=5)
        ax[k].axvline(0, color=INK2, lw=0.9); ax[k].set_title(lab, loc="left"); ax[k].grid(axis="y", visible=False)
        ax[k].set_xlabel("FLARE minus transaction queue")
    ax[0].set_yticks(range(len(ylab))); ax[0].set_yticklabels(ylab)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_bootstrap_ci.png")


def scarcity():
    d = read("X2_label_scarcity_summary.csv").sort_values("label_frac"); fr = d.label_frac * 100
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.4))
    axs[0, 0].errorbar(fr, d.node_AP_mean, d.node_AP_std.fillna(0), color=INK2, marker="o", capsize=2)
    axs[0, 0].set_title("(a) transaction PR-AUC", loc="left")
    for tag, name, c, mk in [("flow", "FLARE groups (ours)", BLUE, "o"), ("tx", "Transaction queue", ORANGE, "s"), ("cc", "Connected components", AQUA, "^")]:
        axs[0, 1].errorbar(fr, d[f"{tag}_ill@0.01_mean"], d[f"{tag}_ill@0.01_std"].fillna(0), color=c, marker=mk, capsize=2, label=name,
                           lw=2.2 if tag == "flow" else 1.3)
        axs[1, 0].errorbar(fr, d[f"{tag}_cases100@0.01_mean"], d[f"{tag}_cases100@0.01_std"].fillna(0), color=c, marker=mk, capsize=2,
                           lw=2.2 if tag == "flow" else 1.3)
    axs[0, 1].set_title("(b) illicit found, 1% budget", loc="left"); axs[1, 0].set_title("(c) cases per 100, 1% budget", loc="left")
    for a in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        a.set_xscale("log"); a.set_xticks(list(fr)); a.set_xticklabels([f"{int(f)}%" for f in fr]); a.minorticks_off()
        a.set_xlabel("training labels used")
    axs[1, 1].axis("off"); h, l = axs[0, 1].get_legend_handles_labels(); axs[1, 1].legend(h, l, loc="center", fontsize=8.5)
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    save(fig, "fig_label_scarcity.png")


def enrichment():
    d = read("E4_group_proxy_metrics.csv")
    lab = d.risk.str.replace("GraphSAGE", "GSAGE") + "\n" + d.method.str.replace("flow(ours) | ours", "FLARE", regex=False) \
        .str.replace("conncomp | sum", "conn. comp.", regex=False).str.replace("ego1hop | mean", "1-hop", regex=False)
    x = np.arange(len(d)); wd = 0.38
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.bar(x - wd / 2, d.purity_known, wd, color=BLUE, label="top-10 groups per step")
    ax.bar(x + wd / 2, d.null_purity, wd, color="#bdbcb7", label="size-matched random groups")
    for i, r in d.iterrows():
        ax.text(i - wd / 2, r.purity_known + 0.02, f"{r.enrichment:.0f}x", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("illicit share of\nlabelled members"); ax.set_ylim(0, 1.1)
    ax.legend(ncol=2, loc="lower right", bbox_to_anchor=(1, 1.0)); ax.grid(axis="x", visible=False)
    save(fig, "fig_enrichment.png")


def counterfactual():
    d = read("E5_counterfactual_summary.csv"); x = np.arange(len(d)); wd = 0.2
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for i, (c, lab, col) in enumerate([("d_top", "gradient-top", VIOLET), ("d_heuristic", "top-risk heuristic", ORANGE),
                                       ("d_random", "random", "#bdbcb7"), ("d_bottom", "gradient-bottom", AQUA)]):
        ax.bar(x + (i - 1.5) * wd, d[c], wd, color=col, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"m = {int(m)}" for m in d.m]); ax.set_xlabel("edges deleted")
    ax.set_ylabel("drop in group score"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    save(fig, "fig_counterfactual.png")


def case_study():
    base = os.path.join(RES, "case_study_group_1")
    if not os.path.exists(base + "_edges.csv"):
        print("case study edges not found; skipped"); return
    m = pd.read_csv(base + ".csv"); e = pd.read_csv(base + "_edges.csv"); lay = pd.read_csv(base + "_layout.csv")
    pos = {r.txId: (r.x, r.y) for r in lay.itertuples()}
    G = nx.DiGraph(); G.add_nodes_from(m.txId)
    G.add_edges_from(e[e.kind == "flow"][["src_txId", "dst_txId"]].itertuples(index=False, name=None))
    S = nx.Graph(e[e.kind == "shared_address"][["src_txId", "dst_txId"]].itertuples(index=False, name=None))
    lab = dict(zip(m.txId, m.label)); risk = dict(zip(m.txId, m.risk))
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    col = [ILLICIT if lab[v] == 1 else (LICIT if lab[v] == 0 else UNKNOWN) for v in G.nodes]
    nx.draw_networkx_nodes(G, pos, node_color=col, node_size=420, edgecolors="black", linewidths=0.5, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=10, width=1.1, node_size=420, ax=ax)
    if S.number_of_edges():
        nx.draw_networkx_edges(S, pos, style="dashed", edge_color=BLUE, width=1.0, ax=ax)
    nx.draw_networkx_labels(G, pos, {v: f"{risk[v]:.2f}" for v in G.nodes}, font_size=8, ax=ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", mfc=c, mec="k", ms=7, label=l) for c, l in
                       [(ILLICIT, "illicit"), (UNKNOWN, "unlabelled")]] +
              [Line2D([0], [0], c="k", label="money flow"), Line2D([0], [0], c=BLUE, ls="--", label="shared address")],
              loc="lower left", fontsize=8)
    ax.axis("off")
    save(fig, "fig_case_study_group_1.png")


# =============================================================================================== diagrams
def box(ax, x, y, w, h, text, fc="white", ec=INK2, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.06", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, fontweight="bold" if bold else "normal",
            linespacing=1.2)


def arrow(ax, p, q, color=INK2, ls="-", rad=0.0, lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


def pipeline():
    fig, ax = plt.subplots(figsize=(W, 7.0)); ax.set_xlim(0, 6); ax.set_ylim(-0.5, 6.6); ax.axis("off")
    rows = [  # stage label, boxes (text, colour)
        ("1  Data", [("203,769 transactions\n182 features", "#eef4fc", BLUE), ("234,355 flow edges\nwithin one step", "#eef4fc", BLUE),
                     ("1.31 M address links\n822,942 addresses", "#eef4fc", BLUE)]),
        ("2  Features", [("Leakage audit of official\nwallet features (lifetime)", "#fff7e6", YELLOW),
                         ("As-of address features\ncurrent + earlier steps,\ntraining labels only", "#eaf7f1", AQUA)]),
        ("3  Risk", [("LightGBM\nPR-AUC 0.750", "white", INK2), ("Direction-aware\nGraphSAGE", "white", INK2),
                     ("Wallet model\n+ tx-risk transfer", "white", INK2)]),
        ("4  Cases", [("Flow groups\nseeds, h hops,\nhub filter, cap 40", "#eef4fc", BLUE),
                      ("Group score\nsum of risk\n(+ motif, exposure)", "#eef4fc", BLUE),
                      ("Review queue\nB = b% of step", "#eef4fc", BLUE)]),
        ("5  Output", [("Ranked cases\nsame yield,\n~60% fewer cases", "#eaf7f1", GREEN), ("Counterfactual\nevidence edges", "#f1effc", VIOLET),
                       ("Evaluation\nPR-AUC, cases,\nbootstrap CIs", "white", INK2)]),
    ]
    rh, gap, top = 0.98, 0.32, 6.5
    for i, (stage, boxes) in enumerate(rows):
        y = top - (i + 1) * rh - i * gap
        ax.text(0.02, y + rh / 2, stage, fontsize=8.5, fontweight="bold", va="center")
        n = len(boxes); x0, span = 1.05, 4.9; bw = (span - (n - 1) * 0.12) / n
        for j, (t, fc, ec) in enumerate(boxes):
            box(ax, x0 + j * (bw + 0.12), y, bw, rh, t, fc=fc, ec=ec, bold=(stage == "5  Output" and j == 0))
        if i < len(rows) - 1:
            arrow(ax, (3.5, y - 0.02), (3.5, y - gap + 0.02))
    ax.text(0.02, -0.45, "All features are point-in-time; lifetime wallet features\nare used only as a labelled comparator.", fontsize=8, color=INK2)
    save(fig, "pipeline.png")


def temporal_split():
    fig, ax = plt.subplots(figsize=(W, 2.5)); ax.set_xlim(0.5, 50.2); ax.set_ylim(-2.3, 3.0); ax.axis("off")
    for t in range(1, 50):
        fc, ec = ("#cfe0f5", BLUE) if t <= 30 else (("#ffe4b8", YELLOW) if t <= 35 else ("#d7f0e4", AQUA))
        ax.add_patch(Rectangle((t, 1.0), 0.85, 0.85, fc=fc, ec=ec, lw=0.5))
        if t in (1, 10, 20, 30, 35, 43, 49):
            ax.text(t + 0.42, 0.78, str(t), ha="center", va="top", fontsize=8, color=INK2)
    ax.text(15.5, 2.1, "train 1-30", ha="center", fontsize=8.5, fontweight="bold", color=BLUE)
    ax.text(33.4, 2.1, "val.", ha="center", fontsize=8.5, fontweight="bold", color="#b77b00")
    ax.text(42.5, 2.1, "test 36-49", ha="center", fontsize=8.5, fontweight="bold", color="#0e7d56")
    ax.plot([43, 43], [0.9, 1.95], color=RED, lw=1.2, ls="--")
    t0 = 38.4
    ax.annotate("", xy=(t0, 0.2), xytext=(t0, 0.95), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
    ax.text(t0, 0.1, "score at step t", fontsize=8, ha="center", va="top")
    ax.annotate("", xy=(1.0, -0.95), xytext=(t0, -0.95), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.8))
    ax.text(1.0, -0.7, "as-of features: current and earlier steps; label history from training steps only", fontsize=8, color="#0e7d56")
    ax.annotate("", xy=(49.9, -1.95), xytext=(1.0, -1.95), arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=1.8))
    ax.text(1.0, -1.7, "official wallet features: lifetime totals copied to every step (include the future)", fontsize=8, color=RED)
    save(fig, "temporal_split.png")


def group_construction():
    G = nx.DiGraph()
    flowA, flowB = [(0, 1), (1, 2), (2, 3), (4, 2)], [(10, 11), (10, 12), (11, 13), (12, 13)]
    hub = 20
    G.add_edges_from(flowA + flowB + [(hub, k) for k in range(21, 29)] + [(3, hub), (13, hub)])
    pos = {0: (0.3, 2.4), 1: (1.0, 2.2), 2: (1.7, 2.5), 3: (2.3, 2.0), 4: (1.2, 3.0), 10: (4.0, 2.6), 11: (4.7, 3.0),
           12: (4.7, 2.2), 13: (5.4, 2.6), hub: (3.0, 1.0)}
    for i, k in enumerate(range(21, 29)):
        ang = np.pi * (1.1 + 0.8 * i / 7); pos[k] = (3.0 + 1.7 * np.cos(ang), 1.0 + 0.75 * np.sin(ang))
    risk = {v: 0.1 for v in G}; risk.update({0: .97, 1: .88, 2: .99, 3: .84, 4: .72, 10: .95, 11: .81, 12: .79, 13: .9, hub: .62})
    seeds, grown = [0, 2, 10], [1, 3, 4, 11, 12, 13]
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.4))
    titles = ["(a) risk scores in one step", "(b) seeds: top-risk transactions", "(c) expansion skips the hub", "(d) resulting cases"]
    cmap = matplotlib.colormaps["Blues"]
    for k, ax in enumerate(axs.flat):
        ax.set_title(titles[k], loc="left"); ax.axis("off"); ax.set_xlim(-0.3, 5.9); ax.set_ylim(-0.25, 3.5)
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c9c8c3", arrowsize=6, width=0.7, node_size=90)
        if k == 0:
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=90, node_color=[cmap(0.2 + 0.8 * risk[v]) for v in G], edgecolors=INK2, linewidths=0.4)
            ax.text(3.0, -0.2, "hub (exchange)", ha="center", fontsize=8, color=INK2)
            continue
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=90, node_color="#e9e8e4", edgecolors="#bdbcb7", linewidths=0.4)
        nx.draw_networkx_nodes(G, pos, nodelist=seeds, ax=ax, node_size=130, node_color=ORANGE, edgecolors=INK, linewidths=0.7)
        if k >= 2:
            nx.draw_networkx_nodes(G, pos, nodelist=grown, ax=ax, node_size=100, node_color="#f7b48f", edgecolors=INK2, linewidths=0.5)
            nx.draw_networkx_edges(G, pos, edgelist=flowA + flowB, ax=ax, edge_color=ORANGE, width=1.5, arrowsize=8, node_size=100)
            ax.plot(*pos[hub], marker="x", ms=9, mew=1.8, color=RED)
            ax.text(3.0, -0.2, "hub skipped (degree > 30)", ha="center", fontsize=8, color=RED)
        if k == 3:
            for cx, cy, w, h, lab in [(1.3, 2.5, 2.6, 1.45, "case 1: 5 tx"), (4.7, 2.6, 2.0, 1.3, "case 2: 4 tx")]:
                ax.add_patch(Ellipse((cx, cy), w, h, fill=False, ec=BLUE, lw=1.3, ls="--"))
                ax.text(cx, cy + h / 2 + 0.05, lab, ha="center", fontsize=8, color=BLUE, fontweight="bold")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    save(fig, "group_construction.png")


def queue_concept():
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    for ax, t in zip(axs, ["(a) transaction queue", "(b) FLARE group queue"]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 5.4); ax.axis("off"); ax.set_title(t, loc="left")
    pts = [(1.0, 4.2), (3.2, 4.5), (5.4, 4.0), (7.6, 4.4), (9.2, 3.6), (1.6, 2.2), (4.0, 2.5), (5.2, 2.5), (7.2, 1.9), (8.8, 1.3)]
    lab = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    axs[0].annotate("", xy=pts[7], xytext=pts[6], arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, shrinkA=6, shrinkB=6))
    for i, (x, y) in enumerate(pts):
        axs[0].scatter(x, y, s=70, color=ILLICIT if lab[i] else UNKNOWN, edgecolor=INK2, lw=0.6, zorder=3)
    axs[0].text(5, 0.25, "10 reviews -> 9 separate cases", ha="center", fontsize=8, color=INK2)
    for (cx, cy), n, lb in [((2.0, 3.5), 4, [1, 1, 0, 1]), ((6.8, 3.6), 3, [1, 0, 0]), ((4.4, 1.6), 3, [1, 1, 0])]:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3
        xs, ys = cx + 0.85 * np.cos(ang), cy + 0.6 * np.sin(ang)
        for j in range(n - 1):
            axs[1].annotate("", xy=(xs[j + 1], ys[j + 1]), xytext=(xs[j], ys[j]),
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1, shrinkA=5, shrinkB=5))
        axs[1].scatter(xs, ys, s=70, color=[ILLICIT if l else UNKNOWN for l in lb], edgecolor=INK2, lw=0.6, zorder=3)
        axs[1].add_patch(Ellipse((cx, cy), 2.7, 1.9, fill=False, ec=BLUE, lw=1.2, ls="--"))
    axs[1].text(5, 0.25, "10 reviews -> 3 linked cases", ha="center", fontsize=8, color=INK2)
    save(fig, "queue_concept.png")


def wireframe():
    fig, ax = plt.subplots(figsize=(W, 3.5)); ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")
    ax.add_patch(Rectangle((0.05, 0.05), 11.9, 7.9, fc="white", ec=INK2, lw=1))
    ax.add_patch(Rectangle((0.05, 7.1), 11.9, 0.85, fc="#e7e6e2", ec=INK2, lw=0.8))
    ax.text(0.3, 7.52, "FLARE · Case Review Console", fontsize=9, fontweight="bold", va="center")
    for i, t in enumerate(["Step 41", "Budget 1%", "LightGBM"]):
        box(ax, 6.2 + i * 1.9, 7.25, 1.75, 0.55, t, fs=8)
    ax.add_patch(Rectangle((0.2, 0.25), 3.0, 6.7, fc="#f4f4f2", ec=MUTED, lw=0.7))
    ax.text(0.35, 6.55, "CASE QUEUE", fontsize=8, fontweight="bold", color=INK2)
    for i, (n, sc) in enumerate([(12, 9.8), (5, 4.6), (4, 3.9), (9, 3.1), (3, 2.7)]):
        y = 5.55 - i * 1.0; sel = i == 0
        ax.add_patch(Rectangle((0.32, y), 2.76, 0.85, fc="#e3edf9" if sel else "white", ec=BLUE if sel else LINE, lw=0.9))
        ax.text(0.45, y + 0.55, f"Case {i + 1}", fontsize=8, fontweight="bold")
        ax.text(0.45, y + 0.18, f"{n} tx · {sc}", fontsize=8, color=INK2)
        ax.add_patch(Rectangle((2.1, y + 0.33), 0.9 * sc / 9.8, 0.2, fc=BLUE, ec="none"))
    ax.add_patch(Rectangle((3.35, 0.25), 4.9, 6.7, fc="white", ec=MUTED, lw=0.7))
    ax.text(3.5, 6.55, "CASE GRAPH", fontsize=8, fontweight="bold", color=INK2)
    P = {0: (4.0, 5.5), 1: (5.0, 4.9), 2: (6.1, 5.4), 3: (7.4, 4.8), 4: (5.4, 3.5), 5: (6.6, 3.2), 6: (7.6, 2.0), 7: (4.2, 2.6)}
    for a, b in [(0, 1), (1, 2), (2, 3), (4, 2), (4, 5), (5, 6), (7, 4)]:
        hl = (a, b) in [(1, 2), (4, 2)]
        arrow(ax, P[a], P[b], color=VIOLET if hl else INK2, lw=2 if hl else 0.9)
    for k, (x, y) in P.items():
        ax.add_patch(plt.Circle((x, y), 0.26, fc=ILLICIT if k in (0, 2, 5) else UNKNOWN, ec=INK2, lw=0.6, zorder=4))
    ax.text(3.5, 0.5, "purple = evidence edges", fontsize=8, color=VIOLET)
    ax.add_patch(Rectangle((8.4, 0.25), 3.4, 6.7, fc="#f4f4f2", ec=MUTED, lw=0.7))
    ax.text(8.55, 6.55, "EVIDENCE", fontsize=8, fontweight="bold", color=INK2)
    for i, (k, v) in enumerate([("Case score", "9.8"), ("Flow motifs", "merge"), ("Bad addresses", "2 / 9"),
                                ("Drop: evidence", "-0.61"), ("Drop: random", "-0.27"), ("Labelled", "3 / 12")]):
        ax.text(8.55, 5.85 - i * 0.85, k, fontsize=8, color=INK2)
        ax.text(11.65, 5.85 - i * 0.85, v, fontsize=8, fontweight="bold", ha="right")
    box(ax, 8.55, 0.45, 1.5, 0.6, "Escalate", fc=BLUE, ec=BLUE, fs=8)
    ax.texts[-1].set_color("white")
    box(ax, 10.15, 0.45, 1.5, 0.6, "Dismiss", ec=BLUE, fs=8)
    save(fig, "wireframe.png")


if __name__ == "__main__":
    labels(); illicit_rate(); models(); per_step(); views(); queues("LightGBM"); queues("GraphSAGE"); bootstrap()
    scarcity(); enrichment(); counterfactual(); case_study()
    pipeline(); temporal_split(); group_construction(); queue_concept(); wireframe()
