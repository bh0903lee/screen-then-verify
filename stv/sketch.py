"""The per-state metric sketch: exact local updates, first-order spectral update, fixed-partition modularity, sampled path length."""
import time
import numpy as np
import networkx as nx
from networkx.algorithms.community import modularity as nx_modularity

import graph_env as base
import operators as P

KEYS = base.KEYS
IDX = {k: i for i, k in enumerate(KEYS)}
TOL = np.array([base.TOL[k] for k in KEYS], np.float64)
SKETCH_DIM = 6


def fiedler_data(G, n):
    deg = np.array([G.degree(v) for v in range(n)], np.float64)
    if n < 3 or G.number_of_edges() == 0 or (deg == 0).any():
        return np.zeros(n), 0.0, 0.0
    if n > LARGE_N:
        vals, vecs = _smallest_eigs(G, n)
    else:
        Lc = nx.normalized_laplacian_matrix(G, nodelist=range(n)).toarray().astype(np.float64)
        vals, vecs = np.linalg.eigh(Lc)
    u = vecs[:, 1]; lam2 = float(max(vals[1], 0.0)); gap = float(vals[2] - vals[1]) if n > 2 else 0.0
    y = u / np.sqrt(deg)
    return y, lam2, gap


LARGE_N = 1500


def _smallest_eigs(G, n):
    from scipy.sparse.linalg import eigsh
    L = nx.normalized_laplacian_matrix(G, nodelist=range(n)).astype(np.float64).tocsc()
    try:
        vals, vecs = eigsh(L, k=3, sigma=-1e-6, which="LM", tol=1e-10)
        o = np.argsort(vals); vals, vecs = vals[o], vecs[:, o]
        if abs(vals[0]) <= 1e-6:
            return vals, vecs
    except Exception:
        pass
    vals, vecs = np.linalg.eigh(L.toarray())
    return vals[:3], vecs[:, :3]


def dlam2_first_order(op, y, lam2):
    rem, add = P.edge_lists(op)
    g = lambda i, j: (y[i] - y[j]) ** 2 - lam2 * (y[i] ** 2 + y[j] ** 2)
    return float(sum(g(i, j) for i, j in add) - sum(g(i, j) for i, j in rem))


def dlam2_combinatorial(op, u):
    rem, add = P.edge_lists(op)
    return float(sum((u[i] - u[j]) ** 2 for i, j in add) - sum((u[i] - u[j]) ** 2 for i, j in rem))


def cnm_partition(G):
    return P.community_labels(G)


def modularity_fixed(G, comm):
    groups = {}
    for v, c in comm.items():
        groups.setdefault(c, set()).add(v)
    return float(nx_modularity(G, list(groups.values())))


def apl_per_source(G, sources):
    out = []
    for s in sources:
        dist = nx.single_source_shortest_path_length(G, s)
        out.append(sum(dist.values()) / (len(dist) - 1) if len(dist) > 1 else 0.0)
    return out


def sampled_apl(G, sources):
    v = apl_per_source(G, sources)
    return float(sum(v) / max(len(v), 1))


def state_sketch(G, n, rng, k_src=8, comm=None):
    t0 = time.perf_counter()
    y, lam2, gap = fiedler_data(G, n); t1 = time.perf_counter()
    if comm is None:
        comm = cnm_partition(G)
    t2 = time.perf_counter()
    k = min(k_src, n); sources = rng.sample(range(n), k) if k < n else list(range(n))
    apl_src = apl_per_source(G, sources); ell_hat = float(sum(apl_src) / max(len(apl_src), 1)); t3 = time.perf_counter()
    clu = nx.clustering(G); C = float(sum(clu.values()) / n)
    Q = modularity_fixed(G, comm); t4 = time.perf_counter()
    return dict(y=y, lam2=lam2, gap=gap, comm=comm, sources=sources, apl_src=apl_src, ell_hat=ell_hat, clu=clu, C=C, Q=Q, n=n,
                stats=P.degree_stats(G), t_fied=t1 - t0, t_comm=t2 - t1, t_bfs=t3 - t2, t_rest=t4 - t3, t_total=t4 - t0)


def cand_deltas(G, op, sk, res, k_src=None, lam_scale=1.0):
    t0 = time.perf_counter(); n = sk["n"]
    rem, add = P.edge_lists(op)
    dlam = lam_scale * dlam2_first_order(op, sk["y"], sk["lam2"])
    dr = float(P.local_deltas(G, op, sk["stats"])["dr"])
    aff_pre = set()
    for (u, v) in rem + add:
        aff_pre |= {u, v} | (set(G.neighbors(u)) & set(G.neighbors(v)))
    P.apply_op(G, op)
    try:
        aff = set(aff_pre)
        for (u, v) in rem + add:
            aff |= set(G.neighbors(u)) & set(G.neighbors(v))
        clu1 = nx.clustering(G, nodes=list(aff))
        dC = float(sum(clu1[v] - sk["clu"][v] for v in aff) / n)
        dQ = modularity_fixed(G, sk["comm"]) - sk["Q"]
        kk = len(sk["sources"]) if k_src is None else min(k_src, len(sk["sources"]))
        srcs = sk["sources"][:kk]; base_ell = float(sum(sk["apl_src"][:kk]) / max(kk, 1))
        if nx.is_connected(G):
            dell = sampled_apl(G, srcs) - base_ell
        else:
            lcc = G.subgraph(max(nx.connected_components(G), key=len))
            s2 = [s for s in srcs if s in lcc] or list(lcc.nodes)[:1]
            dell = sampled_apl(lcc, s2) - base_ell
    finally:
        P.revert_op(G, op)
    d = np.array([dC, dell, dQ, dlam, dr], np.float64)
    res_hat = np.asarray(res, np.float64) + d / TOL
    if not base.ACTIVE.all():
        res_hat = res_hat * base.ACTIVE
    return dict(dC=dC, dell=dell, dQ=dQ, dlam2=dlam, dr=dr, d=d, res_hat=res_hat, W_hat=float(np.abs(res_hat).max()),
                t_sketch=time.perf_counter() - t0)


def sketch_features(cd):
    f = np.clip(cd["d"] / TOL, -10, 10).astype(np.float32)
    return np.concatenate([f, np.array([min(cd["W_hat"], 20.0)], np.float32)])
