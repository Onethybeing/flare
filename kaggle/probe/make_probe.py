import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nbtools import write_nb, write_meta
here = os.path.dirname(os.path.abspath(__file__))
cells = [r'''
import glob, os, pandas as pd
files = sorted(glob.glob("/kaggle/input/**/*.csv", recursive=True))
for f in files: print(os.path.getsize(f)//1_000_000, "MB", f)
''', r'''
for f in files:
    df = pd.read_csv(f, nrows=5)
    print("=====", os.path.basename(f), "ncols", df.shape[1])
    print(list(df.columns)[:12], "...", list(df.columns)[-30:])
    print(df.iloc[:3, :8].to_string())
''', r'''
import numpy as np
d = {os.path.basename(f): f for f in files}
tc = pd.read_csv(d["txs_classes.csv"]); print(tc.columns.tolist(), tc.iloc[:,1].value_counts().to_dict(), len(tc))
te = pd.read_csv(d["txs_edgelist.csv"]); print(te.columns.tolist(), len(te))
tf = pd.read_csv(d["txs_features.csv"], usecols=[0,1]); print(tf.columns.tolist(), len(tf), tf.iloc[:,1].min(), tf.iloc[:,1].max())
ts = dict(zip(tf.iloc[:,0], tf.iloc[:,1]))
same = (te.iloc[:,0].map(ts) == te.iloc[:,1].map(ts)).mean(); print("tx edges within same time step:", same)
wc = pd.read_csv(d["wallets_classes.csv"]); print(wc.columns.tolist(), wc.iloc[:,1].value_counts().to_dict(), len(wc), wc.iloc[:,0].nunique())
wf = pd.read_csv(d["wallets_features.csv"], usecols=[0,1]); print(wf.columns.tolist(), len(wf), "unique addr", wf.iloc[:,0].nunique()); print(wf.iloc[:,1].describe())
for n in ["AddrTx_edgelist.csv","TxAddr_edgelist.csv","AddrAddr_edgelist.csv"]:
    e = pd.read_csv(d[n]); print(n, e.columns.tolist(), len(e), e.iloc[:2].to_string())
''', r'''
wfc = pd.read_csv(d["wallets_features.csv"], nrows=3); print(wfc.T.to_string())
tfc = pd.read_csv(d["txs_features.csv"], nrows=2); print(tfc.iloc[:, 160:].T.to_string())
''']
write_nb(cells, f"{here}/probe.ipynb")
write_meta(here, "btp-ellipticpp-probe", "btp-ellipticpp-probe", "probe.ipynb", gpu=False)
