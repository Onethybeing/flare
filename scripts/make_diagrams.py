"""Generate the explanatory diagrams used in the README and the report (deterministic, matplotlib only).

Outputs (docs/assets/): banner.png, pipeline.png, temporal_split.png, group_construction.png,
queue_concept.png, wireframe.png

    python scripts/make_diagrams.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "assets")
os.makedirs(OUT, exist_ok=True)

# palette (validated categorical slots + neutrals)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, LINE, SURF = "#0b0b0b", "#52514e", "#8a8984", "#d9d8d3", "#fcfcfb"
ILLICIT, LICIT, UNKNOWN = "#d64545", "#2f9e57", "#bdbdbd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE})


def box(ax, x, y, w, h, text, fc="#ffffff", ec=INK2, fs=9.5, bold=False, tc=INK, r=0.02, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", wrap=True, linespacing=1.25)


def arrow(ax, p, q, color=INK2, lw=1.4, style="-|>", ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=12, color=color, lw=lw,
                                 linestyle=ls, connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------------------------- banner
def banner():
    rng = np.random.default_rng(7)
    fig = plt.figure(figsize=(12, 3.2), facecolor="#0f1b2d")
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 3.2); ax.axis("off")
    # background transaction graph (right side)
    G = nx.random_geometric_graph(130, 0.17, seed=3)
    pos = {k: (7.35 + 4.5 * v[0], 0.15 + 2.9 * v[1]) for k, v in nx.get_node_attributes(G, "pos").items()}
    for a, b in G.edges:
        ax.plot(*zip(pos[a], pos[b]), color="#2b4466", lw=0.8, zorder=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=14, node_color="#4a6a94", linewidths=0)
    # three highlighted "cases" (flow groups)
    for cx, cy, col in [(8.3, 2.2, ORANGE), (9.9, 1.0, "#ff5a5a"), (11.1, 2.35, YELLOW)]:
        pts = [(cx + rng.normal(0, 0.22), cy + rng.normal(0, 0.18)) for _ in range(6)]
        for i in range(5):
            ax.annotate("", xy=pts[i + 1], xytext=pts[i],
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.6, mutation_scale=9))
        ax.scatter(*zip(*pts), s=55, color=col, edgecolor="white", lw=0.8, zorder=5)
        ax.add_patch(plt.Circle((cx, cy), 0.62, fill=False, ec=col, lw=1.2, ls=(0, (3, 2)), alpha=0.9))
    ax.text(0.45, 2.15, "Flow Groups", fontsize=34, color="white", fontweight="bold")
    ax.text(0.48, 1.55, "From transactions to cases: leakage-safe review queues", fontsize=12.5, color="#cfe0f5")
    ax.text(0.48, 1.12, "for illicit-flow detection on the Elliptic++ Bitcoin graph", fontsize=12.5, color="#cfe0f5")
    ax.text(0.48, 0.62, "LightGBM  ·  direction-aware GraphSAGE  ·  flow-constrained grouping", fontsize=9.5, color="#8fb0d6")
    ax.text(0.48, 0.32, "point-in-time features  ·  counterfactual evidence  ·  fixed-budget evaluation", fontsize=9.5, color="#8fb0d6")
    save(fig, "banner.png")


# --------------------------------------------------------------------------------------- pipeline
def pipeline():
    fig, ax = plt.subplots(figsize=(14.5, 6.6)); ax.set_xlim(0, 14.5); ax.set_ylim(0, 6.6); ax.axis("off")
    C = [0.15, 3.1, 6.15, 8.95, 11.95]; Wd = [2.6, 2.7, 2.45, 2.65, 2.45]
    for x, t in zip(C, ["1  Data (Elliptic++)", "2  Point-in-time features", "3  Risk models", "4  Cases", "5  Outputs"]):
        ax.text(x, 6.3, t, fontsize=11, fontweight="bold", color=INK)
    fs = 8.8
    box(ax, C[0], 4.65, Wd[0], 1.3, "Transactions\n203,769 nodes, 182 features\n49 time steps", fc="#eef4fc", ec=BLUE, fs=fs)
    box(ax, C[0], 3.15, Wd[0], 1.3, "Money-flow edges\n234,355 directed (tx to tx)\nall within one step", fc="#eef4fc", ec=BLUE, fs=fs)
    box(ax, C[0], 1.65, Wd[0], 1.3, "Address links\n1.31 M address-tx links\n822,942 addresses", fc="#eef4fc", ec=BLUE, fs=fs)
    box(ax, C[0], 0.25, Wd[0], 1.2, "Official wallet features\nlifetime totals per address\n(contain future information)", fc="#fdeeee", ec=RED, fs=fs)
    box(ax, C[1], 4.4, Wd[1], 1.55, "Leakage audit\nconstant lifetime values for\n100% of multi-step addresses;\n10.6% of rows count future steps", fc="#fff7e6", ec=YELLOW, fs=fs)
    box(ax, C[1], 1.85, Wd[1], 2.25, "As-of address features\n- activity in current step\n- history from earlier steps only\n- labels: training period only\nmean / max over inputs and\noutputs of each tx (60 columns)", fc="#eaf7f1", ec=AQUA, fs=fs)
    box(ax, C[1], 0.25, Wd[1], 1.2, "Comparator (XV-lifetime)\nofficial wallet features\naggregated the same way", fc="#fdeeee", ec=RED, fs=fs)
    box(ax, C[2], 4.4, Wd[2], 1.55, "LightGBM\nbest test PR-AUC 0.750", fc="#ffffff", ec=INK2, fs=fs)
    box(ax, C[2], 2.35, Wd[2], 1.6, "Direction-aware\nGraphSAGE\nseparate in / out channels", fc="#ffffff", ec=INK2, fs=fs)
    box(ax, C[2], 0.25, Wd[2], 1.7, "Wallet model\nas-of features plus\ntransaction-risk transfer", fc="#ffffff", ec=INK2, fs=fs)
    box(ax, C[3], 3.95, Wd[3], 2.0, "Flow-constrained groups\nseeds: top-risk transactions\nexpand up to h hops through\nhigh-risk links, skip hubs,\nsplit groups over 40 (Louvain)", fc="#eef4fc", ec=BLUE, fs=fs)
    box(ax, C[3], 2.2, Wd[3], 1.35, "Group score\nsum of risk x (1 + b.motif\n+ g.exposure) x size^(-d)", fc="#eef4fc", ec=BLUE, fs=fs)
    box(ax, C[3], 0.25, Wd[3], 1.55, "Fixed-budget review queue\nB = b% of the step\nwhole groups, then fill", fc="#eef4fc", ec=BLUE, fs=fs)
    box(ax, C[4], 3.95, Wd[4], 2.0, "Evaluation\nPR-AUC, F1, P@K\nillicit found, cases/100\npaired bootstrap CIs", fc="#ffffff", ec=INK2, fs=fs)
    box(ax, C[4], 2.2, Wd[4], 1.35, "Counterfactual\nevidence edges\n(gradient-selected)", fc="#f1effc", ec=VIOLET, fs=fs)
    box(ax, C[4], 0.25, Wd[4], 1.55, "Ranked cases\nsame illicit yield,\nabout 60% fewer cases", fc="#eaf7f1", ec=GREEN, fs=fs, bold=True)
    R = lambda i: C[i] + Wd[i]
    arrow(ax, (R(0), 5.3), (C[1], 3.6)); arrow(ax, (R(0), 3.8), (C[1], 3.2)); arrow(ax, (R(0), 2.3), (C[1], 2.6))
    arrow(ax, (R(0), 0.85), (C[1], 0.85), color=RED, ls="--")
    arrow(ax, (R(1), 3.3), (C[2], 4.9)); arrow(ax, (R(1), 3.0), (C[2], 3.15)); arrow(ax, (R(1), 2.4), (C[2], 1.3))
    arrow(ax, (R(1), 0.85), (C[2], 4.55), color=RED, ls="--", rad=0.3)
    arrow(ax, (R(2), 5.15), (C[3], 5.0)); arrow(ax, (R(2), 3.15), (C[3], 4.3))
    arrow(ax, (C[3] + Wd[3] / 2, 3.95), (C[3] + Wd[3] / 2, 3.55)); arrow(ax, (C[3] + Wd[3] / 2, 2.2), (C[3] + Wd[3] / 2, 1.8))
    arrow(ax, (R(3), 2.85), (C[4], 2.85)); arrow(ax, (R(3), 1.0), (C[4], 1.0))
    ax.text(0.15, 0.0, "Solid arrows: leakage-safe path used for all headline results.   Dashed red: lifetime (future-leaking) comparator.",
            fontsize=8.8, color=INK2)
    save(fig, "pipeline.png")


# ---------------------------------------------------------------------------------- temporal split
def temporal_split():
    fig, ax = plt.subplots(figsize=(12, 3.9)); ax.set_xlim(0, 51); ax.set_ylim(-2.2, 4.3); ax.axis("off")
    for t in range(1, 50):
        col = "#cfe0f5" if t <= 30 else ("#ffe4b8" if t <= 35 else "#d7f0e4")
        ec = BLUE if t <= 30 else (YELLOW if t <= 35 else AQUA)
        ax.add_patch(Rectangle((t, 1.6), 0.9, 0.9, fc=col, ec=ec, lw=0.8))
        if t in (1, 10, 20, 30, 31, 35, 36, 43, 49):
            ax.text(t + 0.45, 1.35, str(t), ha="center", va="top", fontsize=8, color=INK2)
    ax.text(15.5, 2.8, "Train (steps 1-30)", ha="center", fontsize=10.5, fontweight="bold", color=BLUE)
    ax.text(33.5, 2.8, "Validation (31-35)", ha="center", fontsize=10.5, fontweight="bold", color="#b77b00")
    ax.text(42.5, 2.8, "Test (36-49)", ha="center", fontsize=10.5, fontweight="bold", color="#0e7d56")
    ax.plot([43, 43], [1.45, 3.4], color=RED, lw=1.6, ls="--")
    ax.text(43.2, 3.55, "dark-market shutdown (step 43)", fontsize=8.5, color=RED)
    ax.text(0.2, 3.95, "Each time step ≈ two weeks; every money-flow edge joins two transactions of the same step.",
            fontsize=9, color=INK2)
    # as-of vs lifetime at a scoring step t=38
    t0 = 38.45
    ax.annotate("", xy=(t0, 0.75), xytext=(t0, 1.55), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(t0, 0.48, "score at step t", ha="center", fontsize=8.5)
    ax.annotate("", xy=(1.2, -0.25), xytext=(t0 - 0.3, -0.25), arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=2.2))
    ax.text(14, 0.0, "as-of features: current step + strictly earlier steps; label history only from train steps",
            fontsize=8.8, color="#0e7d56")
    ax.annotate("", xy=(49.8, -1.25), xytext=(1.2, -1.25), arrowprops=dict(arrowstyle="<|-|>", color=RED, lw=2.2))
    ax.text(14, -1.0, "official wallet features: lifetime totals copied to every step (include the future → leak)",
            fontsize=8.8, color=RED)
    ax.text(0.2, -2.05, "Scalers, thresholds, group parameters and score weights are fitted on train/validation only; "
                        "the test period is evaluated once.", fontsize=8.5, color=INK2)
    save(fig, "temporal_split.png")


# ------------------------------------------------------------------------------ group construction
def group_construction():
    rng = np.random.default_rng(4)
    # hand-made step graph: two illicit flows, one hub (exchange) and background
    G = nx.DiGraph()
    flowA = [(0, 1), (1, 2), (2, 3), (4, 2)]
    flowB = [(10, 11), (10, 12), (11, 13), (12, 13)]
    hub = 20
    bg = [(hub, k) for k in range(21, 31)] + [(3, hub), (13, hub), (31, 32), (32, 33), (34, 33)]
    G.add_edges_from(flowA + flowB + bg)
    pos = {0: (0.3, 2.4), 1: (1.0, 2.2), 2: (1.7, 2.5), 3: (2.3, 2.0), 4: (1.2, 3.0),
           10: (4.3, 2.6), 11: (5.0, 3.0), 12: (5.0, 2.2), 13: (5.7, 2.6),
           hub: (3.1, 1.0), 31: (5.3, 0.5), 32: (5.9, 0.9), 33: (6.4, 0.5), 34: (6.2, 1.5)}
    for i, k in enumerate(range(21, 31)):
        ang = np.pi * (1.05 + 0.9 * i / 9)
        pos[k] = (3.1 + 1.2 * np.cos(ang) * 1.6, 1.0 + 0.8 * np.sin(ang) + 0.05 * rng.normal())
    risk = {v: 0.1 for v in G}
    risk.update({0: 0.97, 1: 0.88, 2: 0.99, 3: 0.84, 4: 0.72, 10: 0.95, 11: 0.81, 12: 0.79, 13: 0.9, hub: 0.62,
                 21: 0.55, 22: 0.4, 31: 0.3})
    seeds = {2, 10, 0}
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9))
    titles = ["(a) Risk scores on one time step", "(b) Seeds = top-risk transactions",
              "(c) Flow-constrained expansion", "(d) Groups → review cases"]
    cmap = matplotlib.colormaps["Blues"]
    for k, ax in enumerate(axs):
        ax.set_title(titles[k], fontsize=10.5, loc="left")
        ax.axis("off")
        ec = [LINE] * G.number_of_edges()
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#c9c8c3", arrowsize=8, width=0.9, node_size=160)
        if k == 0:
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=160, node_color=[cmap(0.2 + 0.8 * risk[v]) for v in G],
                                   edgecolors=INK2, linewidths=0.5)
            ax.text(3.1, -0.3, "hub (exchange)", ha="center", va="top", fontsize=8.5, color=INK2)
        else:
            base = ["#e9e8e4"] * G.number_of_nodes()
            nx.draw_networkx_nodes(G, pos, ax=ax, node_size=160, node_color=base, edgecolors="#bdbcb7", linewidths=0.5)
            if k >= 1:
                nx.draw_networkx_nodes(G, pos, nodelist=list(seeds), ax=ax, node_size=230, node_color=ORANGE,
                                       edgecolors=INK, linewidths=0.9)
            if k >= 2:
                grown = [1, 3, 4, 11, 12, 13]
                nx.draw_networkx_nodes(G, pos, nodelist=grown, ax=ax, node_size=170, node_color="#f7b48f",
                                       edgecolors=INK2, linewidths=0.6)
                nx.draw_networkx_edges(G, pos, edgelist=flowA + flowB, ax=ax, edge_color=ORANGE, width=2.0,
                                       arrowsize=10, node_size=170)
                ax.plot(*pos[hub], marker="x", ms=13, mew=2.2, color=RED)
                ax.text(3.1, -0.3, "hub skipped (degree > 30)", ha="center", va="top", fontsize=8.5, color=RED)
            if k == 3:
                for cx, cy, w, h, lab in [(1.3, 2.5, 2.6, 1.45, "Case 1: 5 tx"), (5.0, 2.6, 2.0, 1.3, "Case 2: 4 tx")]:
                    ax.add_patch(matplotlib.patches.Ellipse((cx, cy), w, h, fill=False, ec=BLUE, lw=1.8, ls="--"))
                    ax.text(cx, cy + h / 2 + 0.08, lab, ha="center", fontsize=8.5, color=BLUE, fontweight="bold")
        ax.set_xlim(-0.6, 7.0); ax.set_ylim(-0.6, 3.6)
    fig.text(0.01, -0.03, "Connected components of the thresholded graph would merge both flows through the hub into one "
                          "large component; the hub filter, hop limit and size cap keep cases compact.",
             fontsize=9, color=INK2)
    fig.tight_layout()
    save(fig, "group_construction.png")


# ---------------------------------------------------------------------------------- queue concept
def queue_concept():
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, title in zip(axs, ["Transaction queue: top-B transactions", "Group queue: top cases until B"]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off"); ax.set_title(title, fontsize=11, loc="left")
    pts = [(1.0, 3.9), (3.0, 4.2), (5.2, 3.8), (7.4, 4.1), (9.0, 3.5), (1.8, 2.0), (3.9, 2.4), (4.9, 2.4), (6.9, 1.8), (8.6, 1.2)]
    lab = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]
    a_, b_ = pts[6], pts[7]
    axs[0].annotate("", xy=b_, xytext=a_, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=8, shrinkB=8))
    for i, (x, y) in enumerate(pts):
        axs[0].scatter(x, y, s=180, color=ILLICIT if lab[i] else UNKNOWN, edgecolor=INK2, zorder=3)
        if i not in (6, 7):
            axs[0].add_patch(plt.Circle((x, y), 0.45, fill=False, ec=MUTED, lw=1, ls=":"))
    axs[0].add_patch(matplotlib.patches.Ellipse((4.4, 2.4), 1.9, 0.95, fill=False, ec=MUTED, lw=1, ls=":"))
    axs[0].text(5, 0.25, "10 reviews -> 9 separate cases (alerts investigated one by one)", ha="center", fontsize=9, color=INK2)
    groups = [((2.0, 3.3), 4, [1, 1, 0, 1]), ((6.6, 3.4), 3, [1, 0, 0]), ((4.3, 1.55), 3, [1, 1, 0])]
    for (cx, cy), n, lb in groups:
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + 0.3
        xs, ys = cx + 0.8 * np.cos(ang), cy + 0.55 * np.sin(ang)
        for j in range(n - 1):
            axs[1].annotate("", xy=(xs[j + 1], ys[j + 1]), xytext=(xs[j], ys[j]),
                            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2, shrinkA=8, shrinkB=8))
        axs[1].scatter(xs, ys, s=180, color=[ILLICIT if l else UNKNOWN for l in lb], edgecolor=INK2, zorder=3)
        axs[1].add_patch(matplotlib.patches.Ellipse((cx, cy), 2.5, 1.75, fill=False, ec=BLUE, lw=1.6, ls="--"))
    axs[1].text(5, 0.25, "10 reviews -> 3 cases with linked evidence (same budget)", ha="center", fontsize=9, color=INK2)
    fig.text(0.5, 0.0, "red = known illicit, grey = unlabelled;  arrows = money flow;  outline = one investigation case",
             ha="center", fontsize=8.8, color=INK2)
    save(fig, "queue_concept.png")


# ------------------------------------------------------------------------------------- wireframe
def wireframe():
    fig, ax = plt.subplots(figsize=(12, 7)); ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis("off")
    W = "#f4f4f2"
    ax.add_patch(Rectangle((0.05, 0.05), 11.9, 6.9, fc="white", ec=INK2, lw=1.5))
    ax.add_patch(Rectangle((0.05, 6.35), 11.9, 0.6, fc="#e7e6e2", ec=INK2, lw=1))
    ax.text(0.3, 6.65, "Flow Groups · Case Review Console", fontsize=12, fontweight="bold", va="center")
    for i, t in enumerate(["Step 41 ▾", "Budget 1% (37 tx) ▾", "Risk: LightGBM ▾"]):
        box(ax, 6.6 + i * 1.75, 6.45, 1.6, 0.4, t, fc="white", ec=MUTED, fs=8.5)
    # left: case queue
    ax.add_patch(Rectangle((0.2, 0.25), 3.1, 5.95, fc=W, ec=MUTED))
    ax.text(0.35, 5.95, "CASE QUEUE (ranked)", fontsize=9, fontweight="bold", color=INK2)
    for i in range(7):
        y = 5.1 - i * 0.68
        sel = i == 0
        ax.add_patch(Rectangle((0.32, y), 2.86, 0.6, fc="#e3edf9" if sel else "white", ec=BLUE if sel else LINE, lw=1.2))
        ax.text(0.45, y + 0.4, f"Case #{i + 1}", fontsize=8.5, fontweight="bold")
        ax.text(0.45, y + 0.14, f"{[12, 5, 4, 9, 3, 6, 2][i]} tx · score {[9.8, 4.6, 3.9, 3.1, 2.7, 2.2, 1.9][i]}", fontsize=7.5, color=INK2)
        ax.add_patch(Rectangle((2.15, y + 0.22), 0.9 * [1, .55, .45, .38, .33, .27, .22][i], 0.16, fc=BLUE, ec="none"))
    # centre: graph
    ax.add_patch(Rectangle((3.45, 2.35), 5.0, 3.85, fc="white", ec=MUTED))
    ax.text(3.6, 5.95, "CASE GRAPH  (money flow + shared addresses)", fontsize=9, fontweight="bold", color=INK2)
    P = {0: (4.2, 4.9), 1: (5.0, 4.4), 2: (5.9, 4.8), 3: (6.8, 4.3), 4: (5.3, 3.4), 5: (6.4, 3.3), 6: (7.6, 3.6), 7: (4.4, 3.0)}
    E = [(0, 1), (1, 2), (2, 3), (4, 2), (4, 5), (5, 6), (7, 4)]
    for a, b in E:
        hl = (a, b) in [(1, 2), (4, 2)]
        arrow(ax, P[a], P[b], color=VIOLET if hl else INK2, lw=2.4 if hl else 1.2)
    for k, (x, y) in P.items():
        ax.add_patch(plt.Circle((x, y), 0.2, fc=ILLICIT if k in (0, 2, 5) else UNKNOWN, ec=INK2, zorder=4))
    ax.text(3.6, 2.5, "purple = counterfactual evidence edges", fontsize=7.8, color=VIOLET)
    # right: evidence
    ax.add_patch(Rectangle((8.6, 2.35), 3.2, 3.85, fc=W, ec=MUTED))
    ax.text(8.75, 5.95, "EVIDENCE", fontsize=9, fontweight="bold", color=INK2)
    rows = [("Case score", "9.8"), ("Sum of risk", "8.2"), ("Flow motifs", "chain, merge"),
            ("Known-bad addresses", "2 of 9"), ("Top feature", "size"), ("Drop if purple edges removed", "-0.61"),
            ("Drop if random edges removed", "-0.27"), ("Label coverage", "3 / 12")]
    for i, (k, v) in enumerate(rows):
        y = 5.55 - i * 0.4
        ax.text(8.75, y, k, fontsize=7.6, color=INK2, va="top")
        ax.text(11.35, y, v, fontsize=7.8, color=INK, va="top", fontweight="bold", ha="right")
    # bottom: member table + actions
    ax.add_patch(Rectangle((3.45, 0.25), 8.35, 1.95, fc="white", ec=MUTED))
    ax.text(3.6, 1.95, "MEMBER TRANSACTIONS", fontsize=9, fontweight="bold", color=INK2)
    hdr = ["txId", "risk", "label", "in-deg", "out-deg", "addresses", "BTC"]
    for j, h in enumerate(hdr):
        ax.text(3.65 + j * 1.05, 1.62, h, fontsize=7.8, fontweight="bold")
    for i in range(3):
        for j in range(7):
            ax.add_patch(Rectangle((3.62 + j * 1.05, 1.2 - i * 0.32), 0.85, 0.16, fc="#e9e8e4", ec="none"))
    for i, t in enumerate(["Escalate", "Dismiss", "Export"]):
        box(ax, 0.32 + i * 0.98, 0.32, 0.9, 0.38, t, fc=BLUE if i == 0 else "white", ec=BLUE, fs=8,
            tc="white" if i == 0 else BLUE)
    ax.text(6.0, -0.22, "Wireframe of how the case queue would be presented to an AML analyst (concept; not part of the evaluation).",
            ha="center", fontsize=8.5, color=INK2)
    save(fig, "wireframe.png")


if __name__ == "__main__":
    banner(); pipeline(); temporal_split(); group_construction(); queue_concept(); wireframe()
