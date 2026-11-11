# %% [0] setup
QUICK = False
import os, glob, time, json, math, random, warnings, gc
import numpy as np, pandas as pd
import scipy.sparse as sp
warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import networkx as nx
import torch, torch.nn as nn, torch.nn.functional as Fnn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

T0 = time.time()
def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

OUT = "/kaggle/working/results"; FIG = "/kaggle/working/figures"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
SEEDS = [0] if QUICK else [0, 1, 2, 3, 4]
TRAIN_END, VAL_END = 30, 35          # train 1-30, val 31-35, test 36-49
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H_ADDR, H_TX, MAXG = 10, 30, 40      # address-link cap, hub degree, max group size
BUDGETS = [0.005, 0.01, 0.02, 0.05]
SUMMARY = {}
log("device", DEV, "QUICK", QUICK, "lgb", lgb.__version__, "nx", nx.__version__, "torch", torch.__version__)
F = {os.path.basename(f): f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True)}
log(sorted(F))

# %% [1] load transactions, labels, edges
tx = pd.read_csv(F["txs_features.csv"]).rename(columns={"Time step": "ts"})
cls = pd.read_csv(F["txs_classes.csv"])
tx = tx.merge(cls, on="txId", how="left")
tx["y"] = tx["class"].map({1: 1, 2: 0, 3: -1}).astype(int)
tx = tx.sort_values(["ts", "txId"]).reset_index(drop=True)
N = len(tx)
tid2i = pd.Series(np.arange(N), index=tx.txId.values)
LOCAL = [c for c in tx.columns if c.startswith("Local_feature")]
AGG = [c for c in tx.columns if c.startswith("Aggregate_feature")]
AUG = [c for c in tx.columns if c not in LOCAL + AGG + ["txId", "ts", "class", "y"]]
log("tx", tx.shape, "local", len(LOCAL), "agg", len(AGG), "aug", len(AUG))
ts = tx.ts.values; y = tx.y.values
split = np.where(ts <= TRAIN_END, 0, np.where(ts <= VAL_END, 1, 2))
lab = y >= 0
tr, va, te = lab & (split == 0), lab & (split == 1), lab & (split == 2)
log("labeled train/val/test", tr.sum(), va.sum(), te.sum(), "illicit", y[tr].sum(), y[va].sum(), y[te].sum())
E = pd.read_csv(F["txs_edgelist.csv"])
src = tid2i[E.txId1.values].values.astype(np.int64); dst = tid2i[E.txId2.values].values.astype(np.int64)
SUMMARY["audit_tx_edges_same_step"] = float((ts[src] == ts[dst]).mean())
step_nodes = {t: np.where(ts == t)[0] for t in range(1, 50)}
SUMMARY["dataset"] = dict(n_tx=int(N), n_edges=int(len(src)), illicit=int((y == 1).sum()), licit=int((y == 0).sum()),
                          unknown=int((y == -1).sum()), train_lab=int(tr.sum()), val_lab=int(va.sum()), test_lab=int(te.sum()))
log(SUMMARY["dataset"], "same-step edges", SUMMARY["audit_tx_edges_same_step"])

# %% [2] address links + as-of (point-in-time) address features
at = pd.read_csv(F["AddrTx_edgelist.csv"]); ta = pd.read_csv(F["TxAddr_edgelist.csv"])
L = pd.concat([pd.DataFrame({"addr": at.input_address.values, "tx": tid2i.reindex(at.txId.values).values, "role": 0}),
               pd.DataFrame({"addr": ta.output_address.values, "tx": tid2i.reindex(ta.txId.values).values, "role": 1})],
              ignore_index=True)
L = L.dropna(subset=["tx"]).reset_index(drop=True); L["tx"] = L.tx.astype(np.int64)
codes, uniq = pd.factorize(L.addr); L["a"] = codes; ADDR = pd.Index(uniq); L = L.drop(columns="addr")
L["ts"] = ts[L.tx.values]; L["ylab"] = y[L.tx.values]
L["btc"] = tx.total_BTC.values[L.tx.values]; L["fee"] = tx.fees.values[L.tx.values]
L["is_send"] = (L.role == 0).astype(np.int32); L["is_recv"] = (L.role == 1).astype(np.int32)
# labels usable as history: only training-period labels, only from strictly earlier steps (enforced below)
L["ill_tr"] = ((L.ylab == 1) & (L.ts <= TRAIN_END)).astype(np.int32)
L["lic_tr"] = ((L.ylab == 0) & (L.ts <= TRAIN_END)).astype(np.int32)
log("links", len(L), "addresses", len(ADDR))
AS = L.groupby(["a", "ts"]).agg(cur_send=("is_send", "sum"), cur_recv=("is_recv", "sum"), cur_btc=("btc", "sum"),
                                cur_fee=("fee", "sum"), cur_ntx=("tx", "nunique"), ill=("ill_tr", "sum"),
                                lic=("lic_tr", "sum")).reset_index().sort_values(["a", "ts"]).reset_index(drop=True)
g = AS.groupby("a")
for c in ["cur_send", "cur_recv", "cur_btc", "cur_ntx", "ill", "lic"]:
    AS["prior_" + c] = g[c].cumsum() - AS[c]          # strictly earlier steps only
AS["prior_steps"] = g.cumcount()
AS["age"] = AS.ts - g.ts.transform("min")
AS["prior_ill_frac"] = AS.prior_ill / (AS.prior_ill + AS.prior_lic).replace(0, np.nan)
AS["prior_any_ill"] = (AS.prior_ill > 0).astype(np.float32)
AS["prior_mean_btc"] = AS.prior_cur_btc / AS.prior_cur_ntx.replace(0, np.nan)
ASF = ["cur_send", "cur_recv", "cur_btc", "cur_fee", "cur_ntx", "prior_cur_send", "prior_cur_recv", "prior_cur_ntx",
       "prior_steps", "age", "prior_ill", "prior_lic", "prior_ill_frac", "prior_any_ill", "prior_mean_btc"]
LA = L[["tx", "a", "ts", "role"]].merge(AS[["a", "ts"] + ASF], on=["a", "ts"], how="left")

def agg_role(df, cols, prefix):
    out = []
    for r, name in [(0, "in"), (1, "out")]:
        d = df[df.role == r].groupby("tx")[cols].agg(["mean", "max"])
        d.columns = [f"{prefix}_{name}_{c}_{s}" for c, s in d.columns]
        out.append(d)
    return pd.concat(out, axis=1).reindex(np.arange(N)).astype(np.float32)

XV = agg_role(LA, ASF, "xv")
W_tx = np.nan_to_num(np.fmax(XV["xv_in_prior_any_ill_max"].values, XV["xv_out_prior_any_ill_max"].values))
log("XV", XV.shape, "tx with known-bad-history address", int(W_tx.sum()))

# %% [3] official wallet features (lifetime) + leakage audit
wf = pd.read_csv(F["wallets_features.csv"]).rename(columns={"Time step": "ts"})
WCOLS = [c for c in wf.columns if c not in ["address", "ts"]]
wf["a"] = ADDR.get_indexer(wf.address); wf = wf[wf.a >= 0].drop(columns="address").drop_duplicates(["a", "ts"])
wf[WCOLS] = wf[WCOLS].astype(np.float32)
multi = wf.groupby("a").filter(lambda d: len(d) > 1) if QUICK else None
cnt = wf.groupby("a").ts.transform("size")
wm = wf[cnt > 1]
const = wm.groupby("a")[["total_txs", "last_block_appeared_in", "num_timesteps_appeared_in"]].nunique()
chk = wf.merge(AS[["a", "ts", "prior_steps"]], on=["a", "ts"], how="inner")
SUMMARY["audit_wallet_features"] = dict(
    rows=int(len(wf)), addresses_multi_step=int(const.shape[0]),
    frac_multi_step_addr_constant_total_txs=float((const.total_txs == 1).mean()),
    frac_multi_step_addr_constant_last_block=float((const.last_block_appeared_in == 1).mean()),
    frac_rows_counting_future_steps=float((chk.num_timesteps_appeared_in > chk.prior_steps + 1).mean()))
log("wallet leakage audit", SUMMARY["audit_wallet_features"])
LEAKCOLS = ["total_txs", "lifetime_in_blocks", "num_timesteps_appeared_in", "last_block_appeared_in",
            "first_block_appeared_in", "btc_transacted_total", "num_txs_as_sender", "num_txs_as receiver",
            "blocks_btwn_txs_mean", "transacted_w_address_total", "fees_total", "btc_received_total", "btc_sent_total"]
LW = L[["tx", "a", "ts", "role"]].merge(wf[["a", "ts"] + LEAKCOLS], on=["a", "ts"], how="left")
XVL = agg_role(LW, LEAKCOLS, "xvl")
del LW; gc.collect()

# %% [4] feature sets + model helpers
BASE = tx[LOCAL + AGG + AUG].astype(np.float32)
FS = {"LF": tx[LOCAL].astype(np.float32), "AF": tx[LOCAL + AGG].astype(np.float32), "AF+AUG": BASE,
      "AF+AUG+XV": pd.concat([BASE, XV], axis=1), "AF+AUG+XVlifetime": pd.concat([BASE, XVL], axis=1)}

def prep(df):  # signed log + train-standardise, for LR / neural models
    X = df.values.astype(np.float64)
    X = np.sign(X) * np.log1p(np.abs(X))
    mu = np.nanmean(X[split == 0], 0); sd = np.nanstd(X[split == 0], 0) + 1e-6
    X = np.clip(np.nan_to_num((X - mu) / sd), -10, 10)
    return X.astype(np.float32)

def best_thr(yv, pv):
    p, r, t = precision_recall_curve(yv, pv); f = 2 * p * r / (p + r + 1e-12)
    return t[np.nanargmax(f[:-1])]

def node_metrics(p):
    thr = best_thr(y[va], p[va]); yt, pt = y[te], p[te]
    o = dict(val_AP=average_precision_score(y[va], p[va]), AP=average_precision_score(yt, pt),
             AUC=roc_auc_score(yt, pt), F1=f1_score(yt, pt >= thr))
    order = np.argsort(-pt)
    for K in (100, 500):
        o[f"P@{K}"] = float(yt[order[:K]].mean())
    o["F1_pre43"] = f1_score(y[te & (ts < 43)], p[te & (ts < 43)] >= thr)
    o["F1_post43"] = f1_score(y[te & (ts >= 43)], p[te & (ts >= 43)] >= thr)
    return o, thr

LGB_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=20,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=600 if QUICK else 3000, verbose=-1, n_jobs=-1)

def fit_lgb(X, yy, trm, vam, seed, predict_rows=None):
    m = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
    m.fit(X[trm], yy[trm], eval_set=[(X[vam], yy[vam])], eval_metric="average_precision",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m.predict_proba(X if predict_rows is None else X[predict_rows])[:, 1], m

# %% [5] E1: transaction-level models
E1 = []; PRED = {}
def record(model, fs, seed, p):
    o, thr = node_metrics(p); o.update(model=model, features=fs, seed=seed); E1.append(o); PRED[(model, fs, seed)] = p
    log(f"E1 {model:10s} {fs:14s} s{seed} AP={o['AP']:.3f} F1={o['F1']:.3f} P@100={o['P@100']:.2f}")

for fs, df in FS.items():
    X = df.values
    for s in SEEDS:
        p, m = fit_lgb(X, y, tr, va, s); record("LightGBM", fs, s, p)
        if fs == "AF+AUG+XV" and s == 0:
            imp = pd.Series(m.booster_.feature_importance("gain"), index=df.columns).sort_values(ascending=False)
            imp.head(30).to_csv(f"{OUT}/E1_lgbm_xv_top_features.csv")
for fs in ["AF", "AF+AUG", "AF+AUG+XV"]:
    X = np.nan_to_num(FS[fs].values)
    for s in SEEDS[:3]:
        m = RandomForestClassifier(n_estimators=100 if QUICK else 300, max_features=50, n_jobs=-1, random_state=s)
        m.fit(X[tr], y[tr]); record("RF", fs, s, m.predict_proba(X)[:, 1])
    Xp = prep(FS[fs])
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced"); m.fit(Xp[tr], y[tr])
    record("LogReg", fs, 0, m.predict_proba(Xp)[:, 1])

# %% [6] E1b: direction-aware GraphSAGE (own implementation, edge-weighted mean aggregation)
class DirSAGE(nn.Module):
    def __init__(s, d, h=128, graph=True, directed=True, p=0.3):
        super().__init__(); s.graph, s.directed, s.p = graph, directed, p
        k = (3 if directed else 2) if graph else 1
        s.l1 = nn.Linear(d * k, h); s.l2 = nn.Linear(h * k, h); s.out = nn.Linear(h, 1)

    def mix(s, h, S, D, w):
        if not s.graph:
            return h
        n = h.shape[0]
        def agg(a, b):  # mean of messages a -> b
            o = torch.zeros_like(h).index_add_(0, b, h[a] * w[:, None])
            dg = torch.zeros(n, device=h.device).index_add_(0, b, w)
            return o / (dg[:, None] + 1e-9)
        if s.directed:
            return torch.cat([h, agg(S, D), agg(D, S)], 1)
        o = torch.zeros_like(h).index_add_(0, D, h[S] * w[:, None]).index_add_(0, S, h[D] * w[:, None])
        dg = torch.zeros(n, device=h.device).index_add_(0, D, w).index_add_(0, S, w)
        return torch.cat([h, o / (dg[:, None] + 1e-9)], 1)

    def forward(s, x, S, D, w):
        h = Fnn.dropout(Fnn.relu(s.l1(s.mix(x, S, D, w))), s.p, s.training)
        h = Fnn.dropout(Fnn.relu(s.l2(s.mix(h, S, D, w))), s.p, s.training)
        return s.out(h).squeeze(-1)

St, Dt = torch.tensor(src, device=DEV), torch.tensor(dst, device=DEV)
Wt = torch.ones(len(src), device=DEV)

def fit_gnn(Xp, seed, graph=True, directed=True):
    torch.manual_seed(seed); np.random.seed(seed)
    x = torch.tensor(Xp, device=DEV)
    m = DirSAGE(x.shape[1], graph=graph, directed=directed).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=5e-4)
    trI = torch.tensor(np.where(tr)[0], device=DEV); yt = torch.tensor(y[tr], dtype=torch.float32, device=DEV)
    pw = torch.tensor(math.sqrt((y[tr] == 0).sum() / (y[tr] == 1).sum()), device=DEV)
    best, bad, state = -1, 0, None
    for ep in range(60 if QUICK else 500):
        m.train(); opt.zero_grad()
        loss = Fnn.binary_cross_entropy_with_logits(m(x, St, Dt, Wt)[trI], yt, pos_weight=pw)
        loss.backward(); opt.step()
        if ep % 5 == 0:
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(x, St, Dt, Wt)).cpu().numpy()
            ap = average_precision_score(y[va], p[va])
            if ap > best:
                best, bad, state = ap, 0, {k: v.clone() for k, v in m.state_dict().items()}
            else:
                bad += 5
            if bad >= 60:
                break
    m.load_state_dict(state); m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(x, St, Dt, Wt)).cpu().numpy()
    return p, m

XP = {fs: prep(FS[fs]) for fs in ["AF+AUG", "AF+AUG+XV"]}
GNN0 = None
for name, fs, graph, directed in [("MLP", "AF+AUG", False, True), ("SAGE-undir", "AF+AUG", True, False),
                                  ("SAGE-dir", "AF+AUG", True, True), ("SAGE-dir", "AF+AUG+XV", True, True)]:
    for s in SEEDS:
        p, m = fit_gnn(XP[fs], s, graph, directed); record(name, fs, s, p)
        if name == "SAGE-dir" and fs == "AF+AUG+XV" and s == 0:
            GNN0 = m
E1df = pd.DataFrame(E1); E1df.to_csv(f"{OUT}/E1_tx_models_all_seeds.csv", index=False)
E1sum = E1df.groupby(["model", "features"]).agg(["mean", "std"]).drop(columns="seed", level=0)
E1sum.columns = [f"{a}_{b}" for a, b in E1sum.columns]; E1sum = E1sum.sort_values("AP_mean", ascending=False)
E1sum.round(4).to_csv(f"{OUT}/E1_tx_models_summary.csv")
print(E1sum[["AP_mean", "AP_std", "F1_mean", "F1_std", "P@100_mean", "F1_pre43_mean", "F1_post43_mean"]].round(3).to_string())

# %% [7] E2: wallet (actor) level models, as-of vs official (lifetime) features
def ens(model, fs):
    return np.mean([PRED[(model, fs, s)] for s in SEEDS], 0)
R_LGB = ens("LightGBM", "AF+AUG+XV")
R_GNN = ens("SAGE-dir", "AF+AUG+XV")
# out-of-fold tx risk for training steps (so wallet training rows never see in-sample tx predictions)
oof = R_LGB.copy(); Xx = FS["AF+AUG+XV"].values
for f in np.array_split(np.arange(1, TRAIN_END + 1), 5):
    trm = tr & ~np.isin(ts, f); rows = np.isin(ts, f)
    p, _ = fit_lgb(Xx, y, trm, va, 0, predict_rows=rows); oof[rows] = p
L["risk"] = oof[L.tx.values]
rk = L.groupby(["a", "ts"]).risk.agg(["max", "mean"]).rename(columns={"max": "txrisk_max", "mean": "txrisk_mean"})
W = AS.merge(rk, left_on=["a", "ts"], right_index=True, how="left").merge(wf, on=["a", "ts"], how="left")
wcls = pd.read_csv(F["wallets_classes.csv"]); wcls["a"] = ADDR.get_indexer(wcls.address)
amap = pd.Series(wcls["class"].map({1: 1, 2: 0, 3: -1}).values, index=wcls.a.values)
amap = amap[amap.index >= 0]
W["y"] = W.a.map(amap).fillna(-1).astype(int)
wy = W.y.values; wts = W.ts.values; wlab = wy >= 0
wtr, wva, wte = wlab & (wts <= TRAIN_END), wlab & (wts > TRAIN_END) & (wts <= VAL_END), wlab & (wts > VAL_END)
first_ts = W.groupby("a").ts.transform("min").values
log("wallet rows", len(W), "labeled train/val/test", wtr.sum(), wva.sum(), wte.sum())
WFS = {"W-asof": ASF, "W-asof+txrisk": ASF + ["txrisk_max", "txrisk_mean"],
       "W-official(lifetime)": WCOLS, "W-official+txrisk(lifetime)": WCOLS + ["txrisk_max", "txrisk_mean"]}
E2 = []; WPRED = {}
for name, cols in WFS.items():
    Xw = W[cols].values.astype(np.float32)
    for s in SEEDS[:3]:
        p, _ = fit_lgb(Xw, wy, wtr, wva, s); WPRED[(name, s)] = p
        thr = best_thr(wy[wva], p[wva])
        ad = pd.DataFrame({"a": W.a.values[wte], "y": wy[wte], "p": p[wte], "new": first_ts[wte] > VAL_END})
        adr = ad.groupby("a").agg(y=("y", "first"), p=("p", "max"), new=("new", "first"))
        nw = adr[adr.new]
        o = dict(features=name, seed=s, row_AP=average_precision_score(wy[wte], p[wte]),
                 row_F1=f1_score(wy[wte], p[wte] >= thr), addr_AP=average_precision_score(adr.y, adr.p),
                 newaddr_AP=average_precision_score(nw.y, nw.p) if nw.y.sum() > 0 else np.nan,
                 n_new_addr=int(len(nw)))
        E2.append(o); log("E2", o)
E2df = pd.DataFrame(E2); E2df.to_csv(f"{OUT}/E2_wallet_models_all_seeds.csv", index=False)
E2sum = E2df.groupby("features").agg(["mean", "std"]).drop(columns="seed", level=0)
E2sum.columns = [f"{a}_{b}" for a, b in E2sum.columns]; E2sum.round(4).to_csv(f"{OUT}/E2_wallet_models_summary.csv")
print(E2sum.round(3).to_string())
W["wrisk"] = np.mean([WPRED[("W-asof+txrisk", s)] for s in SEEDS[:3]], 0)

# %% [8] step graphs: flow edges + same-address (same step) links
La = L[["a", "ts", "tx"]].drop_duplicates().sort_values(["a", "ts", "tx"])
La["gs"] = La.groupby(["a", "ts"]).tx.transform("size")
Lc = La[(La.gs >= 2) & (La.gs <= H_ADDR)]
nxt = Lc.groupby(["a", "ts"]).tx.shift(-1); mk = nxt.notna()
asrc, adst = Lc.tx[mk].values.astype(np.int64), nxt[mk].astype(np.int64).values
def csr(a, b):
    M = sp.coo_matrix((np.ones(len(a), dtype=np.float32), (a, b)), shape=(N, N)).tocsr(); M.sum_duplicates()
    M.data[:] = 1; return M
A_und = csr(np.r_[src, dst, asrc, adst], np.r_[dst, src, adst, asrc]); A_dir = csr(src, dst)
IP, IX = A_und.indptr, A_und.indices
deg = np.diff(IP)
def nbrs(v):
    return IX[IP[v]:IP[v + 1]]
addr_txs = {k: v.values for k, v in La.groupby(["a", "ts"]).tx}   # (address, step) -> its txs
log("same-address links", len(asrc), "hubs(deg>H_TX)", int((deg > H_TX).sum()))

def components(nodeset):
    nodeset = set(nodeset); seen = set(); comps = []
    for s0 in nodeset:
        if s0 in seen:
            continue
        comp, stack = [], [s0]; seen.add(s0)
        while stack:
            v = stack.pop(); comp.append(v)
            for u in nbrs(v):
                if u in nodeset and u not in seen:
                    seen.add(u); stack.append(u)
        comps.append(comp)
    return comps

def split_big(c, seedset):
    G = nx.Graph(); G.add_nodes_from(c); cs = set(c)
    for v in c:
        G.add_edges_from((v, u) for u in nbrs(v) if u in cs)
    parts = nx.community.louvain_communities(G, seed=0)
    return [list(p) for p in parts if set(p) & seedset]

# %% [9] group builders, features, queues
def seeds_of(t, r, seed_frac):
    nodes = step_nodes[t]; k = max(1, int(round(seed_frac * len(nodes))))
    return nodes[np.argsort(-r[nodes])][:k]

def flow_groups(t, r, seed_frac, q_exp, hops, hub_filter=True, size_cap=True):
    nodes = step_nodes[t]; seeds = seeds_of(t, r, seed_frac); seedset = set(seeds.tolist())
    tau = np.quantile(r[nodes], 1 - q_exp); U = set(seedset)
    for s0 in seeds:
        if hub_filter and deg[s0] > H_TX:
            continue
        frontier = [s0]
        for _ in range(hops):
            nf = []
            for v in frontier:
                for u in nbrs(v):
                    if u not in U and r[u] >= tau and (not hub_filter or deg[u] <= H_TX):
                        U.add(u); nf.append(u)
            frontier = nf
    out = []
    for c in components(U):
        if not (set(c) & seedset):
            continue
        out += split_big(c, seedset) if (size_cap and len(c) > MAXG) else [c]
    return [np.array(sorted(g)) for g in out]

def cc_groups(t, r, seed_frac, q_exp, hops=None):
    nodes = step_nodes[t]; seeds = seeds_of(t, r, seed_frac); seedset = set(seeds.tolist())
    tau = np.quantile(r[nodes], 1 - q_exp)
    U = set(nodes[r[nodes] >= tau].tolist()) | seedset
    return [np.array(sorted(c)) for c in components(U) if set(c) & seedset]

def ego_groups(t, r, seed_frac, q_exp=None, hops=None):
    return [np.array(sorted(set([s0]) | set(nbrs(s0).tolist()))) for s0 in seeds_of(t, r, seed_frac)]

def gfeat(gr, r):
    n = len(gr); rs = np.sort(r[gr])[::-1]
    sub = A_dir[gr][:, gr]
    ind = np.asarray(sub.sum(0)).ravel(); outd = np.asarray(sub.sum(1)).ravel()
    motif = ((outd >= 2).sum() + (ind >= 2).sum() + ((ind == 1) & (outd == 1)).sum()) / n if n > 1 else 0.0
    return np.array([n, rs[0], rs.mean(), rs[:3].mean(), rs.sum(), motif, W_tx[gr].mean()])
# feature index: 0 n, 1 max, 2 mean, 3 top3, 4 sum, 5 motif, 6 known-bad-address exposure

BASEIDX = {"max": 1, "mean": 2, "top3": 3, "sum": 4}
def score(Fm, kind, w=(0, 0, 0, "sum")):
    if kind in BASEIDX:
        return Fm[:, BASEIDX[kind]]
    b, gm, d, base = w   # ours: base x (1 + b*motif + g*known-bad exposure) x size^-d
    return Fm[:, BASEIDX[base]] * (1 + b * Fm[:, 5] + gm * Fm[:, 6]) * Fm[:, 0] ** (-d)

def run_queue(groups, sc, r, t, B):
    nodes = step_nodes[t]; seen, rev = set(), []
    for gi in np.argsort(-sc):
        new = [v for v in groups[gi] if v not in seen]
        if not new:
            continue
        if len(rev) + len(new) > B:
            new = sorted(new, key=lambda v: -r[v])[:B - len(rev)]
        rev += new; seen.update(new)
        if len(rev) >= B:
            break
    if len(rev) < B:
        for v in nodes[np.argsort(-r[nodes])]:
            if v not in seen:
                rev.append(v); seen.add(v)
            if len(rev) >= B:
                break
    return np.array(rev)

def qstats(rev, t):
    yy = y[rev]; comps = components(rev)
    return np.array([(yy == 1).sum(), (yy == 0).sum(), (yy == -1).sum(), (y[step_nodes[t]] == 1).sum(), len(comps),
                     sum(1 for c in comps if (y[c] == 1).any()), len(rev)])
QCOLS = ["ill", "lic", "unk", "tot_ill", "cases", "ill_cases", "reviewed"]

def wallet_queue(t, r, B):
    Wt_ = W[W.ts == t].sort_values("wrisk", ascending=False)
    seen, rev = set(), []
    for a in Wt_.a.values:
        new = [v for v in addr_txs.get((a, t), []) if v not in seen]
        if len(rev) + len(new) > B:
            new = sorted(new, key=lambda v: -r[v])[:B - len(rev)]
        rev += new; seen.update(new)
        if len(rev) >= B:
            break
    return np.array(rev)

def pooled(st):
    s = st.sum(0); d = dict(zip(QCOLS, s))
    return dict(illicit_found=int(d["ill"]), precision_known=d["ill"] / max(d["ill"] + d["lic"], 1),
                recall=d["ill"] / max(d["tot_ill"], 1), label_coverage=(d["ill"] + d["lic"]) / d["reviewed"],
                cases_per_100=100 * d["cases"] / d["reviewed"], illicit_per_case=d["ill"] / max(d["cases"], 1),
                illicit_cases=int(d["ill_cases"]))

# %% [10] E3/E4: tune group methods on validation steps, evaluate on test steps
VAL_STEPS = list(range(TRAIN_END + 1, VAL_END + 1)); TEST_STEPS = list(range(VAL_END + 1, 50))
# seeds scale with the review budget: seed_frac = sfm * budget
GRID = dict(sfm=[1.0] if QUICK else [0.5, 1.0, 1.5], q_exp=[0.1] if QUICK else [0.05, 0.1, 0.2], hops=[1, 2])
WGRID = [(b, gm, d, base) for base in ["sum", "top3"] for b in [0, 0.25, 0.5, 1.0] for gm in [0, 0.5, 1.0]
         for d in [0, 0.25, 0.5]]
BUILDERS = {"flow(ours)": flow_groups, "conncomp": cc_groups, "ego1hop": ego_groups}

def build_all(builder, r, steps, cfg, bud, **kw):
    out = {}
    sf = cfg["sfm"] * bud
    for t in steps:
        gs = builder(t, r, sf, cfg["q_exp"], cfg["hops"], **kw) if builder is flow_groups else builder(t, r, sf, cfg["q_exp"])
        out[t] = (gs, np.array([gfeat(g_, r) for g_ in gs]) if gs else np.zeros((0, 7)))
    return out

def queue_rows(built, r, steps, kind, w, bud, full=True):
    rows = []
    for t in steps:
        gs, Fm = built[t]; B = max(1, int(math.ceil(bud * len(step_nodes[t]))))
        rev = run_queue(gs, score(Fm, kind, w), r, t, B) if len(gs) else run_queue([], np.array([]), r, t, B)
        rows.append(qstats(rev, t) if full else np.array([(y[rev] == 1).sum()]))
    return np.array(rows)

def tune(builder, r, kinds):
    """pick (cfg, kind, w) maximising illicit found on validation steps, summed over all budgets"""
    best = (-1e9, None)
    for sfm in GRID["sfm"]:
        for qe in GRID["q_exp"]:
            for hp in (GRID["hops"] if builder is flow_groups else [1]):
                cfg = dict(sfm=sfm, q_exp=qe, hops=hp)
                builts = {bud: build_all(builder, r, VAL_STEPS, cfg, bud) for bud in BUDGETS}
                for kind in kinds:
                    for w in (WGRID if kind == "ours" else [(0, 0, 0, "sum")]):
                        o = sum(queue_rows(builts[bud], r, VAL_STEPS, kind, w, bud, full=False).sum() /
                                max(1, sum((y[step_nodes[t]] == 1).sum() for t in VAL_STEPS)) for bud in BUDGETS)
                        if o > best[0] + 1e-12:
                            best = (o, dict(cfg=cfg, kind=kind, w=w))
    return best[1]

E3rows, CFGS, STEPSTATS, TESTBUILT = [], {}, {}, {}
for rname, r in [("LightGBM", R_LGB), ("GraphSAGE", R_GNN)]:
    for bname, builder in BUILDERS.items():
        kinds = ["max", "mean", "top3", "sum"] + (["ours"] if bname == "flow(ours)" else [])
        CFGS[(rname, bname)] = tune(builder, r, kinds); log("tuned", rname, bname, CFGS[(rname, bname)])
    cf = CFGS[(rname, "flow(ours)")]
    for bud in BUDGETS:
        tb = {b: build_all(BUILDERS[b], r, TEST_STEPS, CFGS[(rname, b)]["cfg"], bud) for b in BUILDERS}
        TESTBUILT[(rname, bud)] = tb
        variants = {"tx-queue (no groups)": None, "wallet-queue": "wallet"}
        for b in BUILDERS:
            variants[f"{b} | {CFGS[(rname, b)]['kind']}"] = (tb[b], CFGS[(rname, b)]["kind"], CFGS[(rname, b)]["w"])
        for k in ["max", "mean", "top3", "sum"]:
            variants[f"flow(ours) | {k} (ablation: plain score)"] = (tb["flow(ours)"], k, (0, 0, 0, "sum"))
        variants["flow(ours) | no hub filter (ablation)"] = (build_all(flow_groups, r, TEST_STEPS, cf["cfg"], bud, hub_filter=False), cf["kind"], cf["w"])
        variants["flow(ours) | no size cap (ablation)"] = (build_all(flow_groups, r, TEST_STEPS, cf["cfg"], bud, size_cap=False), cf["kind"], cf["w"])
        for vname, v in variants.items():
            if v is None:
                st = np.array([qstats(run_queue([], np.array([]), r, t, max(1, int(math.ceil(bud * len(step_nodes[t]))))), t) for t in TEST_STEPS])
            elif v == "wallet":
                st = np.array([qstats(wallet_queue(t, r, max(1, int(math.ceil(bud * len(step_nodes[t]))))), t) for t in TEST_STEPS])
            else:
                st = queue_rows(v[0], r, TEST_STEPS, v[1], v[2], bud)
            STEPSTATS[(rname, vname, bud)] = st
            E3rows.append(dict(risk=rname, queue=vname, budget=bud, **pooled(st)))
    log("E3 done", rname)
E3df = pd.DataFrame(E3rows); E3df.to_csv(f"{OUT}/E3_queues.csv", index=False)
print(E3df[E3df.budget == 0.01].round(3).to_string())
SUMMARY["tuned_configs"] = {f"{k[0]}|{k[1]}": v for k, v in CFGS.items()}

# %% [11] E4: group-level proxy metrics with size-matched random null
rng = np.random.default_rng(0)
def null_group(t, n):
    nodes = step_nodes[t]
    for _ in range(50):
        s0 = rng.choice(nodes); seen, q = {s0}, [s0]
        while q and len(seen) < n:
            v = q.pop(0); nb = nbrs(v).copy(); rng.shuffle(nb)
            for u in nb:
                if u not in seen and len(seen) < n:
                    seen.add(u); q.append(u)
        if len(seen) == n:
            return np.array(list(seen))
    return rng.choice(nodes, n, replace=False)

E4rows = []
for rname, r in [("LightGBM", R_LGB), ("GraphSAGE", R_GNN)]:
    for b in BUILDERS:
        c = CFGS[(rname, b)]; built = TESTBUILT[(rname, 0.01)][b]
        acc = []; nulls = []
        for t in TEST_STEPS:
            gs, Fm = built[t]
            if not len(gs):
                continue
            for gi in np.argsort(-score(Fm, c["kind"], c["w"]))[:10]:
                gg = gs[gi]; yy = y[gg]; ill, kn = (yy == 1).sum(), (yy >= 0).sum()
                acc.append([len(gg), ill >= 1, ill >= 2, ill, kn])
                for _ in range(3 if QUICK else 20):
                    ny = y[null_group(t, len(gg))]; nulls.append([(ny == 1).sum(), (ny >= 0).sum(), (ny == 1).sum() >= 1])
        acc = np.array(acc, dtype=float); nulls = np.array(nulls, dtype=float)
        purity = acc[:, 3].sum() / max(acc[:, 4].sum(), 1); npur = nulls[:, 0].sum() / max(nulls[:, 1].sum(), 1)
        E4rows.append(dict(risk=rname, method=f"{b} | {c['kind']}", n_groups=len(acc), mean_size=acc[:, 0].mean(),
                           median_size=float(np.median(acc[:, 0])), max_size=acc[:, 0].max(),
                           hit_lenient=acc[:, 1].mean(), hit_strict=acc[:, 2].mean(), purity_known=purity,
                           label_coverage=acc[:, 4].sum() / acc[:, 0].sum(), null_purity=npur,
                           null_hit_lenient=nulls[:, 2].mean(), enrichment=purity / max(npur, 1e-9)))
E4df = pd.DataFrame(E4rows); E4df.to_csv(f"{OUT}/E4_group_proxy_metrics.csv", index=False)
print(E4df.round(3).to_string())

# %% [12] E5: counterfactual evidence test (GraphSAGE, gradient-selected evidence edges)
cf = CFGS[("GraphSAGE", "flow(ours)")]; xg = torch.tensor(XP["AF+AUG+XV"], device=DEV)
for p_ in GNN0.parameters():
    p_.requires_grad_(False)
built = TESTBUILT[("GraphSAGE", 0.01)]["flow(ours)"]
bw = cf["w"]; E5rows = []
def gscore_local(prob, gl, motif, wexp, n):
    pg = prob[gl]; top3 = torch.topk(pg, min(3, len(gl))).values.mean()
    kind = cf["kind"]
    if kind == "ours":
        base = pg.sum() if bw[3] == "sum" else top3
        s_ = base * (1 + bw[0] * motif + bw[1] * wexp) * n ** (-bw[2])
    else:
        s_ = {"max": pg.max(), "mean": pg.mean(), "top3": top3, "sum": pg.sum()}[kind]
    return s_, s_
for t in TEST_STEPS:
    nodes = step_nodes[t]; gid = {v: i for i, v in enumerate(nodes)}
    em = ts[src] == t; es, ed = src[em], dst[em]
    S_ = torch.tensor([gid[v] for v in es], device=DEV); D_ = torch.tensor([gid[v] for v in ed], device=DEV)
    xs = xg[torch.tensor(nodes, device=DEV)]
    gs, Fm = built[t]
    if not len(gs):
        continue
    order = [i for i in np.argsort(-score(Fm, cf["kind"], bw)) if len(gs[i]) >= 2][:3 if QUICK else 10]
    for gi in order:
        gg = gs[gi]; gl = torch.tensor([gid[v] for v in gg], device=DEV); n = len(gg)
        wv = torch.ones(len(es), device=DEV, requires_grad=True)
        prob = torch.sigmoid(GNN0(xs, S_, D_, wv))
        S0, top0 = gscore_local(prob, gl, Fm[gi, 5], Fm[gi, 6], n)
        grad = torch.autograd.grad(top0, wv)[0].detach().cpu().numpy()
        cand = np.where(np.abs(grad) > 0)[0]
        if len(cand) < 2:
            continue
        ranked = cand[np.argsort(-grad[cand])]
        gset = set(gg.tolist())
        def after(remove):
            w2 = torch.ones(len(es), device=DEV); w2[torch.tensor(remove, device=DEV, dtype=torch.long)] = 0
            with torch.no_grad():
                pr = torch.sigmoid(GNN0(xs, S_, D_, w2))
            keep = np.ones(len(es), bool); keep[remove] = False
            inner = [(a, b) for a, b, k in zip(es, ed, keep) if k and a in gset and b in gset]
            ind = {v: 0 for v in gg}; outd = {v: 0 for v in gg}
            for a, b in inner:
                outd[a] += 1; ind[b] += 1
            motif = sum((outd[v] >= 2) + (ind[v] >= 2) + (ind[v] == 1 and outd[v] == 1) for v in gg) / n
            s, tp = gscore_local(pr, gl, motif, Fm[gi, 6], n)
            return float(s), float(tp)
        # heuristic baseline: edges touching the highest-risk member
        hub = gg[np.argmax(R_GNN[gg])]; heur_all = [i for i in cand if es[i] == hub or ed[i] == hub]
        for mm in (1, 3, 5):
            if len(cand) < mm:
                continue
            top_s, _ = after(list(ranked[:mm])); bot_s, _ = after(list(ranked[-mm:]))
            rnd = [after(list(np.random.default_rng(k).choice(cand, mm, replace=False)))[0] for k in range(10)]
            heur = after(heur_all[:mm])[0] if len(heur_all) >= 1 else np.nan
            suff = after(list(ranked[mm:]))[0]       # keep only the top-m evidence edges in the receptive field
            E5rows.append(dict(step=t, size=n, m=mm, n_cand=len(cand), S_orig=float(S0), d_top=float(S0) - top_s,
                               d_random=float(S0) - np.mean(rnd), d_bottom=float(S0) - bot_s,
                               d_heuristic=float(S0) - heur, sufficiency=suff / float(S0)))
E5df = pd.DataFrame(E5rows); E5df.to_csv(f"{OUT}/E5_counterfactual_raw.csv", index=False)
E5sum = E5df.groupby("m").agg(groups=("S_orig", "size"), d_top=("d_top", "mean"), d_random=("d_random", "mean"),
                              d_bottom=("d_bottom", "mean"), d_heuristic=("d_heuristic", "mean"),
                              sufficiency=("sufficiency", "mean"))
E5sum["top_beats_random_frac"] = E5df.groupby("m").apply(lambda d: (d.d_top > d.d_random).mean())
E5sum.round(4).to_csv(f"{OUT}/E5_counterfactual_summary.csv"); print(E5sum.round(4).to_string())

# %% [13] E6: per-step curves, bootstrap CIs, figures, summary
def boot_diff(a, b, reps=2000):
    rng_ = np.random.default_rng(1); n = len(a); d = []
    for _ in range(reps):
        i = rng_.integers(0, n, n); d.append(a[i].sum() - b[i].sum())
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
CI = []
for rname in ["LightGBM", "GraphSAGE"]:
    ours = [k for k in STEPSTATS if k[0] == rname and k[1].startswith("flow(ours) |") and "ablation" not in k[1]]
    for bud in BUDGETS:
        o = STEPSTATS[[k for k in ours if k[2] == bud][0]]
        for other in ["tx-queue (no groups)", "wallet-queue"] + [k[1] for k in STEPSTATS if k[0] == rname and k[2] == bud
                                                                  and (k[1].startswith("conncomp") or k[1].startswith("ego1hop"))]:
            b_ = STEPSTATS[(rname, other, bud)]
            lo, hi = boot_diff(o[:, 0], b_[:, 0]); clo, chi = boot_diff(o[:, 4], b_[:, 4])
            CI.append(dict(risk=rname, budget=bud, vs=other, illicit_diff=int(o[:, 0].sum() - b_[:, 0].sum()),
                           illicit_CI95=(lo, hi), cases_diff=int(o[:, 4].sum() - b_[:, 4].sum()), cases_CI95=(clo, chi)))
CIdf = pd.DataFrame(CI); CIdf.to_csv(f"{OUT}/E6_bootstrap_queue_diffs.csv", index=False); print(CIdf.to_string())

per = []
for (model, fs, s), p in PRED.items():
    if s != 0:
        continue
    thr = best_thr(y[va], p[va])
    for t in TEST_STEPS:
        m_ = te & (ts == t)
        if y[m_].sum() > 0:
            per.append(dict(model=model, features=fs, step=t, F1=f1_score(y[m_], p[m_] >= thr),
                            AP=average_precision_score(y[m_], p[m_]), n_illicit=int(y[m_].sum())))
PER = pd.DataFrame(per); PER.to_csv(f"{OUT}/E6_per_step.csv", index=False)

plt.figure(figsize=(8, 4))
for (mdl, fs) in [("LightGBM", "AF+AUG+XV"), ("LightGBM", "AF"), ("RF", "AF"), ("SAGE-dir", "AF+AUG+XV"), ("MLP", "AF+AUG")]:
    d = PER[(PER.model == mdl) & (PER.features == fs)]
    plt.plot(d.step, d.F1, marker="o", ms=3, label=f"{mdl} {fs}")
plt.axvline(43, ls="--", c="gray"); plt.text(43.2, 0.95, "dark-market shutdown", fontsize=8)
plt.xlabel("test time step"); plt.ylabel("illicit F1"); plt.ylim(0, 1); plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(f"{FIG}/fig_per_step_f1.png", dpi=150); plt.close()

for rname in ["LightGBM", "GraphSAGE"]:
    plt.figure(figsize=(6, 4))
    for q in E3df[E3df.risk == rname].queue.unique():
        if "ablation" in q:
            continue
        d = E3df[(E3df.risk == rname) & (E3df.queue == q)]
        plt.plot(d.budget * 100, d.illicit_found, marker="o", label=q)
    plt.xlabel("review budget (% of step's transactions)"); plt.ylabel("illicit transactions found (test)")
    plt.title(f"Review queues ({rname} risk)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(f"{FIG}/fig_queues_{rname}.png", dpi=150); plt.close()

for rname in ["LightGBM", "GraphSAGE"]:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for q in E3df[E3df.risk == rname].queue.unique():
        if "ablation" in q:
            continue
        d = E3df[(E3df.risk == rname) & (E3df.queue == q)]
        ax[0].plot(d.budget * 100, d.cases_per_100, marker="o", label=q)
        ax[1].plot(d.budget * 100, d.illicit_per_case, marker="o", label=q)
    ax[0].set_ylabel("separate cases per 100 reviewed tx (lower = less fragmented)")
    ax[1].set_ylabel("illicit transactions per case")
    for a_ in ax:
        a_.set_xlabel("review budget (% of step's transactions)")
    ax[1].legend(fontsize=7); fig.suptitle(f"Case fragmentation ({rname} risk)"); fig.tight_layout()
    fig.savefig(f"{FIG}/fig_fragmentation_{rname}.png", dpi=150); plt.close(fig)

# case study: highest-scoring flow group with >=4 members in the test period (LightGBM risk, 1% budget)
try:
    cfl = CFGS[("LightGBM", "flow(ours)")]; best_g, best_s, best_t = None, -1, None
    for t in TEST_STEPS:
        gs, Fm = TESTBUILT[("LightGBM", 0.01)]["flow(ours)"][t]
        if not len(gs):
            continue
        sc = score(Fm, cfl["kind"], cfl["w"])
        for i in np.argsort(-sc):
            if len(gs[i]) >= 4 and sc[i] > best_s:
                best_g, best_s, best_t = gs[i], sc[i], t
                break
    if best_g is not None:
        gset = set(best_g.tolist()); G = nx.DiGraph(); G.add_nodes_from(best_g)
        G.add_edges_from((a, b) for a, b in zip(src, dst) if a in gset and b in gset)
        same = [(a, b) for a, b in zip(asrc, adst) if a in gset and b in gset]
        pos = nx.spring_layout(G.to_undirected(), seed=1)
        col = ["#d62728" if y[v] == 1 else ("#2ca02c" if y[v] == 0 else "#bbbbbb") for v in G.nodes]
        plt.figure(figsize=(6, 5))
        nx.draw_networkx_nodes(G, pos, node_color=col, node_size=[200 + 600 * R_LGB[v] for v in G.nodes])
        nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", edge_color="#333333")
        nx.draw_networkx_edges(nx.Graph(same), pos, style="dashed", edge_color="#1f77b4")
        nx.draw_networkx_labels(G, pos, {v: f"{R_LGB[v]:.2f}" for v in G.nodes}, font_size=7)
        plt.title(f"Case study: top flow group, test step {best_t} (n={len(best_g)})\n"
                  "red=illicit, green=licit, grey=unknown; solid=money flow, dashed=shared address; label=risk",
                  fontsize=8)
        plt.axis("off"); plt.tight_layout(); plt.savefig(f"{FIG}/fig_case_study_group.png", dpi=150); plt.close()
        pd.DataFrame({"txId": tx.txId.values[best_g], "risk": R_LGB[best_g], "label": y[best_g]}).to_csv(
            f"{OUT}/case_study_group_members.csv", index=False)
except Exception as ex:
    log("case study figure failed:", ex)

lk = E1sum.reset_index(); lk = lk[(lk.model == "LightGBM")]
plt.figure(figsize=(6, 3.5)); plt.bar(lk.features, lk.AP_mean, yerr=lk.AP_std, color="#4C72B0")
plt.ylabel("test PR-AUC"); plt.title("LightGBM by feature view (XVlifetime = official lifetime wallet features)")
plt.xticks(rotation=20, fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/fig_feature_views.png", dpi=150); plt.close()

if len(E5df):
    d = E5sum.reset_index(); xx = np.arange(len(d)); wdt = 0.2
    plt.figure(figsize=(6, 3.5))
    for i, (c, lbl) in enumerate([("d_top", "gradient-top"), ("d_heuristic", "top-risk heuristic"),
                                  ("d_random", "random"), ("d_bottom", "gradient-bottom")]):
        plt.bar(xx + i * wdt, d[c], wdt, label=lbl)
    plt.xticks(xx + 1.5 * wdt, [f"m={m}" for m in d.m]); plt.ylabel("group-score drop after deletion")
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{FIG}/fig_counterfactual.png", dpi=150); plt.close()

SUMMARY["E1_top"] = E1sum.head(12)[["AP_mean", "AP_std", "F1_mean", "P@100_mean"]].round(4).reset_index().to_dict("records")
SUMMARY["E2"] = E2sum.round(4).reset_index().to_dict("records")
SUMMARY["E4"] = E4df.round(4).to_dict("records")
SUMMARY["E5"] = E5sum.round(4).reset_index().to_dict("records") if len(E5df) else []
def md_table(df, digits=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(digits)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in df.values)
with open(f"{OUT}/tables.md", "w") as f:
    f.write("# Benchmark tables\n\n## E1 Transaction-level models (test steps 36-49, mean over seeds)\n\n")
    f.write(md_table(E1sum[["AP_mean", "AP_std", "AUC_mean", "F1_mean", "F1_std", "P@100_mean", "P@500_mean",
                            "F1_pre43_mean", "F1_post43_mean"]].reset_index()))
    f.write("\n## E2 Wallet-level models\n\n" + md_table(E2sum.reset_index()))
    f.write("\n## Leakage audit\n\n" + md_table(pd.DataFrame([SUMMARY["audit_wallet_features"]])))
    for bud in BUDGETS:
        f.write(f"\n## E3 Review queues at budget {bud*100:g}% per step\n\n" + md_table(E3df[E3df.budget == bud]))
    f.write("\n## E4 Group-level proxy metrics (top-10 groups per test step)\n\n" + md_table(E4df))
    if len(E5df):
        f.write("\n## E5 Counterfactual evidence deletion (GraphSAGE)\n\n" + md_table(E5sum.reset_index(), 4))
    f.write("\n## E6 Bootstrap 95% CI of differences (ours minus other)\n\n" + md_table(CIdf))
SUMMARY["runtime_min"] = (time.time() - T0) / 60
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(SUMMARY, f, indent=1, default=str)
log("DONE", f"{SUMMARY['runtime_min']:.1f} min")
