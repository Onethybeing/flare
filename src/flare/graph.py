"""Transaction graph used for grouping and fragmentation statistics.

Two kinds of links connect transactions *within the same time step*:

* **money flow**: directed edges ``i -> j`` (an output of ``i`` is spent by ``j``); in Elliptic++
  every flow edge joins two transactions of the same step;
* **shared address**: transactions that touch the same address in the same step. To avoid
  quadratic cliques around busy addresses, the transactions of an (address, step) pair are
  chained in id order, and only pairs with at most ``address_cap`` transactions are linked.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def same_address_pairs(addr: np.ndarray, step: np.ndarray, tx: np.ndarray, address_cap: int = 10):
    """Chain the transactions that share an address in the same step.

    Parameters
    ----------
    addr, step, tx : 1-D integer arrays of equal length (one row per address-transaction link).
    address_cap : (address, step) groups with more transactions than this are ignored (hubs).

    Returns
    -------
    (a, b) : arrays of transaction ids; each pair is an undirected shared-address link.
    """
    import pandas as pd

    df = pd.DataFrame({"a": addr, "ts": step, "tx": tx}).drop_duplicates().sort_values(["a", "ts", "tx"])
    df["gs"] = df.groupby(["a", "ts"]).tx.transform("size")
    df = df[(df.gs >= 2) & (df.gs <= address_cap)]
    nxt = df.groupby(["a", "ts"]).tx.shift(-1)
    keep = nxt.notna()
    return df.tx[keep].to_numpy(np.int64), nxt[keep].to_numpy(np.int64)


class StepGraph:
    """Undirected neighbourhood structure plus the directed flow sub-graph.

    Parameters
    ----------
    n : number of transactions (node ids are ``0..n-1``).
    flow_src, flow_dst : directed money-flow edges.
    addr_a, addr_b : optional undirected shared-address links.
    """

    def __init__(self, n: int, flow_src, flow_dst, addr_a=None, addr_b=None):
        flow_src = np.asarray(flow_src, dtype=np.int64)
        flow_dst = np.asarray(flow_dst, dtype=np.int64)
        addr_a = np.asarray([] if addr_a is None else addr_a, dtype=np.int64)
        addr_b = np.asarray([] if addr_b is None else addr_b, dtype=np.int64)
        self.n = int(n)
        self.flow_src, self.flow_dst = flow_src, flow_dst
        rows = np.r_[flow_src, flow_dst, addr_a, addr_b]
        cols = np.r_[flow_dst, flow_src, addr_b, addr_a]
        und = sp.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n)).tocsr()
        und.sum_duplicates()
        und.data[:] = 1
        und.setdiag(0)
        und.eliminate_zeros()
        self.und = und
        self.flow = self._binary(flow_src, flow_dst, n)
        self.indptr, self.indices = und.indptr, und.indices
        self.degree = np.diff(und.indptr)

    @staticmethod
    def _binary(a, b, n):
        m = sp.coo_matrix((np.ones(len(a), dtype=np.float32), (a, b)), shape=(n, n)).tocsr()
        m.sum_duplicates()
        m.data[:] = 1
        return m

    def neighbors(self, v: int) -> np.ndarray:
        """Undirected neighbours (flow in either direction or shared address)."""
        return self.indices[self.indptr[v]:self.indptr[v + 1]]

    def components(self, nodes) -> list[list[int]]:
        """Connected components of the sub-graph induced by ``nodes``."""
        nodeset = set(int(v) for v in nodes)
        seen: set[int] = set()
        comps = []
        for s0 in nodeset:
            if s0 in seen:
                continue
            comp, stack = [], [s0]
            seen.add(s0)
            while stack:
                v = stack.pop()
                comp.append(v)
                for u in self.neighbors(v):
                    u = int(u)
                    if u in nodeset and u not in seen:
                        seen.add(u)
                        stack.append(u)
            comps.append(comp)
        return comps

    def flow_degrees(self, group) -> tuple[np.ndarray, np.ndarray, int]:
        """In-degree, out-degree and edge count of the flow sub-graph induced by ``group``."""
        g = np.asarray(group, dtype=np.int64)
        sub = self.flow[g][:, g]
        indeg = np.asarray(sub.sum(0)).ravel()
        outdeg = np.asarray(sub.sum(1)).ravel()
        return indeg, outdeg, int(sub.nnz)
