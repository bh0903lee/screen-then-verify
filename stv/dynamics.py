"""Spreading dynamics on one graph: Reed-Frost SIR (with calibration of the transmission probability) and the fractional-threshold model."""
import math
import numpy as np
import scipy.sparse as sp

VERSION = "pc_dyn-2026-11-12"
MAJOR_FRAC = 0.05
CASCADE_FRAC = 0.50
KUR_GRID = tuple(float(x) for x in np.round(np.geomspace(0.1, 10.0, 12), 6))


def csr_of_edges(n, edges):
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    r = np.concatenate([e[:, 0], e[:, 1]]); c = np.concatenate([e[:, 1], e[:, 0]])
    A = sp.csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(n, n))
    A.sum_duplicates(); A.sort_indices()
    return A.indptr.astype(np.int64), A.indices.astype(np.int64)


def csr_of(G):
    n = G.number_of_nodes()
    assert set(G.nodes()) == set(range(n)), "nodes must be 0..N−1"
    return csr_of_edges(n, list(G.edges()))


def _nbrs(csr, frontier):
    ip, ix = csr
    if len(frontier) == 1:
        u = int(frontier[0]); return ix[ip[u]:ip[u + 1]]
    starts = ip[frontier]; ends = ip[frontier + 1]; lens = ends - starts
    if lens.sum() == 0:
        return np.zeros(0, np.int64)
    idx = np.repeat(starts - np.cumsum(np.r_[0, lens[:-1]]), lens) + np.arange(lens.sum())
    return ix[idx]


def sir_run(csr, p, rng, seed_node=None):
    ip, _ = csr; n = len(ip) - 1
    state = np.zeros(n, np.int8)
    s0 = int(rng.integers(n)) if seed_node is None else int(seed_node)
    state[s0] = 1; frontier = np.array([s0], np.int64); curve = [1]
    while len(frontier):
        nb = _nbrs(csr, frontier)
        state[frontier] = 2
        nb = nb[state[nb] == 0]
        if len(nb):
            hit = nb[rng.random(len(nb)) < p]
            new = np.unique(hit)
        else:
            new = np.zeros(0, np.int64)
        state[new] = 1; curve.append(len(new)); frontier = new
    if curve[-1] == 0:
        curve.pop()
    return float((state == 2).sum()) / n, np.asarray(curve, np.int64)


def sir_summary(csr, p, runs, rng):
    fin = np.empty(runs); peak = np.full(runs, np.nan)
    for i in range(runs):
        f, c = sir_run(csr, p, rng); fin[i] = f
        if f > MAJOR_FRAC:
            peak[i] = float(np.argmax(c))
    maj = fin > MAJOR_FRAC
    return dict(sir_final=float(fin.mean()), sir_final_sd=float(fin.std(ddof=1)) if runs > 1 else 0.0,
                sir_pmajor=float(maj.mean()), sir_peak=float(np.nanmean(peak)) if maj.any() else float("nan"),
                sir_peak_sd=float(np.nanstd(peak, ddof=1)) if maj.sum() > 1 else float("nan"), sir_runs=int(runs), sir_major_runs=int(maj.sum()))


def mean_final(csr, p, runs, seed):
    rng = np.random.default_rng(seed)
    return float(np.mean([sir_run(csr, p, rng)[0] for _ in range(runs)]))


def calibrate_p(csr, target, rng_seed, runs=5000, tol=0.005, max_iter=30):
    lo, hi = 0.0, 1.0; best = None
    for it in range(1, max_iter + 1):
        p = 0.5 * (lo + hi); f = mean_final(csr, p, runs, rng_seed)
        best = (p, f, it)
        if abs(f - target) <= tol:
            return dict(p=p, mean_final=f, iterations=it, converged=True)
        if f < target:
            lo = p
        else:
            hi = p
    return dict(p=best[0], mean_final=best[1], iterations=best[2], converged=False)


def threshold_cascade(csr, phi, seeds):
    ip, ix = csr; n = len(ip) - 1; deg = np.diff(ip)
    act = np.zeros(n, bool); cnt = np.zeros(n, np.int64)
    need = np.ceil(phi * deg - 1e-12).astype(np.int64)
    need = np.maximum(need, 1)
    stack = []
    for s in np.unique(np.asarray(seeds, np.int64)):
        act[s] = True; stack.append(int(s))
    while stack:
        u = stack.pop()
        for w in ix[ip[u]:ip[u + 1]]:
            if not act[w]:
                cnt[w] += 1
                if cnt[w] >= need[w]:
                    act[w] = True; stack.append(int(w))
    return float(act.sum()) / n


def threshold_runs(csr, phi, seeding, draws, rng):
    ip, ix = csr; n = len(ip) - 1; out = np.empty(draws)
    for i in range(draws):
        if seeding == "ego":
            u = int(rng.integers(n)); seeds = np.r_[u, ix[ip[u]:ip[u + 1]]]
        elif seeding == "uniform":
            seeds = rng.choice(n, size=max(1, math.ceil(0.02 * n)), replace=False)
        else:
            raise ValueError(seeding)
        out[i] = threshold_cascade(csr, phi, seeds)
    return out


def threshold_summary(csr, phi, seeding, draws, rng):
    x = threshold_runs(csr, phi, seeding, draws, rng)
    return dict(adopt=float(x.mean()), adopt_sd=float(x.std(ddof=1)) if draws > 1 else 0.0, pcascade=float((x > CASCADE_FRAC).mean()))


