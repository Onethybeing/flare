# %% [X1] extended tuning grid (fixes grid-edge issue: sfm previously tuned at the edge 1.5)
EXTRA = {}
X = FS["AF+AUG+XV"].values
R_LGB = np.mean([fit_lgb(X, y, tr, va, s)[0] for s in SEEDS], 0)
XPx = prep(FS["AF+AUG+XV"])
R_GNN = np.mean([fit_gnn(XPx, s)[0] for s in SEEDS], 0)
log("risk AP lgb", average_precision_score(y[te], R_LGB[te]), "gnn", average_precision_score(y[te], R_GNN[te]))
np.savez_compressed(f"{OUT}/risk_scores_extra.npz", R_LGB=R_LGB, R_GNN=R_GNN)
GRID = dict(sfm=[1.0, 1.5] if QUICK else [1.0, 1.5, 2.0, 3.0], q_exp=[0.1] if QUICK else [0.05, 0.1, 0.2], hops=[1, 2])

def txq(r, t, bud):
    return qstats(run_queue([], np.array([]), r, t, max(1, int(math.ceil(bud * len(step_nodes[t]))))), t)

X1 = []
for rname, r in [("LightGBM", R_LGB), ("GraphSAGE", R_GNN)]:
    cfgs = {b: tune(BUILDERS[b], r, ["max", "mean", "top3", "sum"] + (["ours"] if b == "flow(ours)" else []))
            for b in ["flow(ours)", "conncomp"]}
    log("X1 tuned", rname, cfgs)
    EXTRA[f"X1_cfg_{rname}"] = cfgs
    for bud in BUDGETS:
        st_tx = np.array([txq(r, t, bud) for t in TEST_STEPS])
        X1.append(dict(risk=rname, queue="tx-queue (no groups)", budget=bud, **pooled(st_tx)))
        for b, c in cfgs.items():
            built = build_all(BUILDERS[b], r, TEST_STEPS, c["cfg"], bud)
            st = queue_rows(built, r, TEST_STEPS, c["kind"], c["w"], bud)
            X1.append(dict(risk=rname, queue=f"{b} | {c['kind']} (sfm={c['cfg']['sfm']})", budget=bud, **pooled(st)))
            if b == "flow(ours)":
                st2 = queue_rows(built, r, TEST_STEPS, "sum", (0, 0, 0, "sum"), bud)
                X1.append(dict(risk=rname, queue="flow(ours) | sum (ablation: plain score)", budget=bud, **pooled(st2)))
X1df = pd.DataFrame(X1); X1df.to_csv(f"{OUT}/X1_extended_grid_queues.csv", index=False)
print(X1df.round(3).to_string())

# %% [X2] label scarcity (training labels subsampled; history-label features rebuilt with the same mask)
def build_xv(keep):
    yl = np.where(keep, y, -1)
    Lq = L[["a", "tx", "ts", "role", "is_send", "is_recv", "btc", "fee"]].copy()
    Lq["ill"] = ((yl[Lq.tx.values] == 1) & (Lq.ts <= TRAIN_END)).astype(np.int32)
    Lq["lic"] = ((yl[Lq.tx.values] == 0) & (Lq.ts <= TRAIN_END)).astype(np.int32)
    A = Lq.groupby(["a", "ts"]).agg(cur_send=("is_send", "sum"), cur_recv=("is_recv", "sum"), cur_btc=("btc", "sum"),
                                    cur_fee=("fee", "sum"), cur_ntx=("tx", "nunique"), ill=("ill", "sum"),
                                    lic=("lic", "sum")).reset_index().sort_values(["a", "ts"]).reset_index(drop=True)
    g_ = A.groupby("a")
    for c in ["cur_send", "cur_recv", "cur_btc", "cur_ntx", "ill", "lic"]:
        A["prior_" + c] = g_[c].cumsum() - A[c]
    A["prior_steps"] = g_.cumcount(); A["age"] = A.ts - g_.ts.transform("min")
    A["prior_ill_frac"] = A.prior_ill / (A.prior_ill + A.prior_lic).replace(0, np.nan)
    A["prior_any_ill"] = (A.prior_ill > 0).astype(np.float32)
    A["prior_mean_btc"] = A.prior_cur_btc / A.prior_cur_ntx.replace(0, np.nan)
    LAq = Lq[["tx", "a", "ts", "role"]].merge(A[["a", "ts"] + ASF], on=["a", "ts"], how="left")
    xv = agg_role(LAq, ASF, "xv")
    w = np.nan_to_num(np.fmax(xv["xv_in_prior_any_ill_max"].values, xv["xv_out_prior_any_ill_max"].values))
    return xv, w

# group configurations tuned in the full run (LightGBM risk); kept fixed here
FULLCFG = {"flow(ours)": (dict(sfm=1.5, q_exp=0.1, hops=2), "ours", (0.5, 0.5, 0.5, "sum")),
           "conncomp": (dict(sfm=1.5, q_exp=0.1, hops=1), "sum", (0, 0, 0, "sum"))}
X2 = []; W_full = W_tx.copy()
for frac in ([1.0, 0.2] if QUICK else [1.0, 0.5, 0.2, 0.1]):
    for rep in range(1 if QUICK else 3):
        rng_ = np.random.default_rng(100 + rep)
        keep = lab.copy(); idx = np.where(tr)[0]
        if frac < 1:
            drop = []
            for cl in (0, 1):
                ic = idx[y[idx] == cl]; drop.append(rng_.choice(ic, int(round(len(ic) * (1 - frac))), replace=False))
            keep[np.concatenate(drop)] = False
        trk = tr & keep
        xv, W_tx = build_xv(keep)
        Xk = pd.concat([BASE, xv], axis=1).values
        r = fit_lgb(Xk, y, trk, va, rep)[0]
        row = dict(label_frac=frac, rep=rep, n_train_labels=int(trk.sum()), n_train_illicit=int(y[trk].sum()),
                   node_AP=average_precision_score(y[te], r[te]))
        for bud in (0.01, 0.02):
            p_tx = pooled(np.array([txq(r, t, bud) for t in TEST_STEPS]))
            row[f"tx_ill@{bud}"] = p_tx["illicit_found"]; row[f"tx_cases100@{bud}"] = p_tx["cases_per_100"]
            for b, (cfg, kind, w) in FULLCFG.items():
                p = pooled(queue_rows(build_all(BUILDERS[b], r, TEST_STEPS, cfg, bud), r, TEST_STEPS, kind, w, bud))
                tag = "flow" if b == "flow(ours)" else "cc"
                row[f"{tag}_ill@{bud}"] = p["illicit_found"]; row[f"{tag}_cases100@{bud}"] = p["cases_per_100"]
        X2.append(row); log("X2", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
W_tx = W_full
X2df = pd.DataFrame(X2); X2df.to_csv(f"{OUT}/X2_label_scarcity_raw.csv", index=False)
X2sum = X2df.drop(columns="rep").groupby("label_frac").agg(["mean", "std"])
X2sum.columns = [f"{a}_{b}" for a, b in X2sum.columns]; X2sum.round(3).to_csv(f"{OUT}/X2_label_scarcity_summary.csv")
print(X2sum.round(3).T.to_string())

# %% [X3] figures + tables
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8)); fr = X2sum.index.values * 100
ax[0].errorbar(fr, X2sum.node_AP_mean, X2sum.node_AP_std.fillna(0), marker="o"); ax[0].set_ylabel("Transaction PR-AUC (test)")
for tag, lbl in [("tx", "Transaction queue"), ("flow", "Flow groups (ours)"), ("cc", "Connected components")]:
    ax[1].errorbar(fr, X2sum[f"{tag}_ill@0.01_mean"], X2sum[f"{tag}_ill@0.01_std"].fillna(0), marker="o", label=lbl)
    ax[2].errorbar(fr, X2sum[f"{tag}_cases100@0.01_mean"], X2sum[f"{tag}_cases100@0.01_std"].fillna(0), marker="o", label=lbl)
ax[1].set_ylabel("Illicit found @1% budget"); ax[2].set_ylabel("Cases per 100 reviewed @1%"); ax[2].legend(fontsize=8)
for a_ in ax:
    a_.set_xscale("log"); a_.set_xticks(fr); a_.set_xticklabels([f"{int(f)}%" for f in fr])
    a_.set_xlabel("Training labels used"); a_.grid(alpha=.3)
fig.suptitle("Label scarcity (LightGBM risk, fixed group configuration)"); fig.tight_layout()
fig.savefig(f"{FIG}/fig_label_scarcity.png", dpi=170); plt.close(fig)

def md_table(df, digits=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(digits)
    return ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" +
            "".join("| " + " | ".join(map(str, r_)) + " |\n" for r_ in df.values))
with open(f"{OUT}/tables_extra.md", "w") as f:
    f.write("# Supplementary tables\n\n## X1 Extended tuning grid (sfm up to 3.0)\n\n" + md_table(X1df))
    f.write("\n## X2 Label scarcity (mean over 3 random label subsets)\n\n" + md_table(X2sum.reset_index()))
with open(f"{OUT}/summary_extra.json", "w") as f:
    json.dump(EXTRA, f, indent=1, default=str)
log("DONE")
