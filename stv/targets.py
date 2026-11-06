"""Synthetic target families (seven generators) and target sampling."""
import sys
from pathlib import Path
import numpy as np, networkx as nx

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import graph_env as base

N = 200; KBAND = (5.5, 6.5); MIN_N = 150
KEYS = base.KEYS
FAMILIES = ["ws", "holme_kim", "sbm", "soft_rgg", "gaussian_part", "relaxed_caveman", "extended_ba"]


def _lcc(G):
    if G.number_of_nodes() == 0:
        return G
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return nx.convert_node_labels_to_integers(G)


def _seed(rng):
    return int(rng.integers(1 << 30))


def g_ws(rng):
    return _lcc(nx.watts_strogatz_graph(N, 6, float(rng.uniform(0.02, 0.6)), seed=_seed(rng)))


def g_holme_kim(rng):
    return _lcc(nx.powerlaw_cluster_graph(N, 3, float(rng.uniform(0.0, 1.0)), seed=_seed(rng)))


def g_sbm(rng):
    b = int(rng.integers(2, 7)); sizes = [N // b] * b; sizes[0] += N - sum(sizes); s = N / b
    ratio = float(rng.uniform(1.5, 20.0)); pout = 6.0 / (ratio * (s - 1) + (N - s)); pin = min(1.0, ratio * pout)
    P = np.full((b, b), pout); np.fill_diagonal(P, pin)
    return _lcc(nx.stochastic_block_model(sizes, P, seed=_seed(rng)))


def g_soft_rgg(rng):
    r = float(np.sqrt(6.0 / (N * np.pi)) * rng.uniform(0.95, 1.25))
    return _lcc(nx.soft_random_geometric_graph(N, r, seed=_seed(rng)))


def g_gaussian_part(rng):
    s = int(rng.integers(20, 60)); v = float(rng.uniform(5, 20))
    pin = float(rng.uniform(0.05, 0.16)); pout = float(rng.uniform(0.004, 0.018))
    return _lcc(nx.gaussian_random_partition_graph(N, s, v, pin, pout, seed=_seed(rng)))


def g_relaxed_caveman(rng):
    return _lcc(nx.relaxed_caveman_graph(N // 7, 7, float(rng.uniform(0.1, 0.5)), seed=_seed(rng)))


def g_extended_ba(rng):
    return _lcc(nx.extended_barabasi_albert_graph(N, 3, float(rng.uniform(0.0, 0.15)), float(rng.uniform(0.0, 0.15)),
                                                  seed=_seed(rng)))


GEN = {"ws": g_ws, "holme_kim": g_holme_kim, "sbm": g_sbm, "soft_rgg": g_soft_rgg,
       "gaussian_part": g_gaussian_part, "relaxed_caveman": g_relaxed_caveman, "extended_ba": g_extended_ba}


def exact_spectral_gap(G):
    L = nx.normalized_laplacian_matrix(G, nodelist=range(G.number_of_nodes())).toarray().astype(np.float64)
    vals = np.linalg.eigvalsh(L)
    return float(max(0.0, vals[1]))


def target_from(G):
    m = base.compute_metrics(G, KEYS); d = {k: round(float(m[k]), 6) for k in KEYS}
    d["spectral_gap"] = round(exact_spectral_gap(G), 6)
    return None if any(not np.isfinite(v) for v in d.values()) else d


def sample_one(family, rng, kband=KBAND, tries=80):
    for _ in range(tries):
        try:
            G = GEN[family](rng)
        except Exception:
            continue
        if G is None or G.number_of_nodes() < MIN_N or not nx.is_connected(G):
            continue
        kb = 2 * G.number_of_edges() / G.number_of_nodes()
        if not (kband[0] <= kb <= kband[1]):
            continue
        t = target_from(G)
        if t is not None:
            t["_kbar_src"] = round(kb, 3); t["_n_src"] = G.number_of_nodes()
            t["_degseq"] = sorted((d for _, d in G.degree()), reverse=True)
            t["_graph_edges"] = [(int(u), int(v)) for u, v in G.edges()]
            return t
    return None


def sample_targets(n, seed, families=FAMILIES, kband=KBAND, with_degseq=False, with_graph=False):
    rng = np.random.default_rng(seed + 1); out = []; i = 0; fails = 0
    while len(out) < n and fails < 10 * len(families):
        fam = families[i % len(families)]; i += 1
        t = sample_one(fam, rng, kband)
        if t is None:
            fails += 1; continue
        d = {k: t[k] for k in KEYS}
        if with_degseq:
            d["degree_seq"] = t["_degseq"]
        if with_graph:
            d["_graph_edges"] = t["_graph_edges"]; d["_n"] = t["_n_src"]
        out.append((d, fam))
    if len(out) < n:
        raise RuntimeError(f"mix7 sampler: only {len(out)}/{n} targets after {fails} failures")
    return out


