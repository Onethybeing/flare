import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import write_nb, write_meta
here = os.path.dirname(os.path.abspath(__file__))
code = open(os.path.join(here, "..", "main", "btp_main.py"), encoding="utf-8").read()
sec = {m.group(1): None for m in re.finditer(r"(?m)^# %% \[(\d+)\]", code)}
parts = re.split(r"(?m)^(?=# %% \[\d+\])", code)
bynum = {re.match(r"# %% \[(\d+)\]", p).group(1): p for p in parts if p.startswith("# %% [")}
cells = [bynum["0"], bynum["1"], bynum["2"], "XVL = XV  # lifetime view not needed here\n" + bynum["4"], r'''
# risk scores (same recipe as the full run: LightGBM AF+AUG+XV, 5 seeds)
X = FS["AF+AUG+XV"].values
R_LGB = np.mean([fit_lgb(X, y, tr, va, s)[0] for s in SEEDS], 0)
np.savez_compressed(f"{OUT}/risk_scores.npz", txId=tx.txId.values, ts=ts, y=y, R_LGB=R_LGB)
log("risk AP test", average_precision_score(y[te], R_LGB[te]))
''', bynum["8"], bynum["9"], r'''
CFG = dict(sfm=1.5, q_exp=0.1, hops=2); W_ = (0.5, 0.5, 0.5, "sum"); bud = 0.01
cands = []
for t in range(VAL_END + 1, 50):
    gs = flow_groups(t, R_LGB, CFG["sfm"] * bud, CFG["q_exp"], CFG["hops"])
    if not gs: continue
    Fm = np.array([gfeat(g_, R_LGB) for g_ in gs]); sc = score(Fm, "ours", W_)
    for i in np.argsort(-sc)[:10]:
        g_ = gs[i]; yy = y[g_]
        cands.append((t, g_, sc[i], int((yy == 1).sum()), int((yy == -1).sum()), A_dir[g_][:, g_].nnz))
# readable example: 6-14 members, with money-flow edges, mixed labels if possible
pick = sorted([c for c in cands if 6 <= len(c[1]) <= 14 and c[5] >= 3],
              key=lambda c: (-(c[3] > 0 and c[4] > 0), -c[2]))
log("candidate groups", len(cands), "readable", len(pick))
for k, (t, g_, s_, ill, unk, ne) in enumerate(pick[:2]):
    gset = set(g_.tolist()); G = nx.DiGraph(); G.add_nodes_from(g_)
    G.add_edges_from((a, b) for a, b in zip(src, dst) if a in gset and b in gset)
    same = [(a, b) for a, b in zip(asrc, adst) if a in gset and b in gset]
    pos = nx.kamada_kawai_layout(nx.Graph(list(G.edges) + same) if (G.number_of_edges() + len(same)) else G)
    for v in g_:
        pos.setdefault(v, np.random.rand(2))
    col = ["#d62728" if y[v] == 1 else ("#2ca02c" if y[v] == 0 else "#bdbdbd") for v in G.nodes]
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    nx.draw_networkx_nodes(G, pos, node_color=col, node_size=650, edgecolors="black", linewidths=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=14, width=1.4, node_size=650, ax=ax)
    if same:
        nx.draw_networkx_edges(nx.Graph(same), pos, style="dashed", edge_color="#1f77b4", width=1.2, ax=ax)
    nx.draw_networkx_labels(G, pos, {v: f"{R_LGB[v]:.2f}" for v in G.nodes}, font_size=7, ax=ax)
    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], marker="o", ls="", mfc=c_, mec="k", ms=9, label=l_) for c_, l_ in
           [("#d62728", "illicit"), ("#2ca02c", "licit"), ("#bdbdbd", "unknown")]] + \
          [Line2D([0], [0], c="k", label="money flow (tx -> tx)"), Line2D([0], [0], c="#1f77b4", ls="--", label="shared address (same step)")]
    ax.legend(handles=leg, fontsize=7, loc="lower left")
    ax.set_title(f"Flow group from test step {t}: {len(g_)} transactions, {ill} illicit, {unk} unlabelled\n"
                 "node label = model risk score", fontsize=9)
    ax.axis("off"); fig.tight_layout(); fig.savefig(f"{FIG}/fig_case_study_group_{k+1}.png", dpi=170); plt.close(fig)
    pd.DataFrame({"txId": tx.txId.values[g_], "risk": R_LGB[g_], "label": y[g_]}).to_csv(f"{OUT}/case_study_group_{k+1}.csv", index=False)
log("DONE")
''']
cells = [c.replace("# %% ", "# ", 1) for c in cells]
write_nb(cells, os.path.join(here, "polish.ipynb"))
write_meta(here, "btp-flowgroup-figures", "btp-flowgroup-figures", "polish.ipynb", gpu=False)
print("ok")
