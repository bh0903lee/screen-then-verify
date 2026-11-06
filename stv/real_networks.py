"""Real-network loading, preprocessing and targets."""
import io
import random
import sys
import zipfile
import gzip
from pathlib import Path
import numpy as np, networkx as nx

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import graph_env as base
import targets as PT

DATA = _REPO / "data" / "real"
KEYS = base.KEYS
NAMES = {"jazz": "jazz_collab", "celegans": "celegansneural", "netscience": "netscience", "polbooks": "polbooks",
         "football": "football", "email": "email-Eu-core"}


def _edges_from_zip(path):
    import csv
    with zipfile.ZipFile(path) as z:
        txt = z.read("edges.csv").decode()
    out = []
    for rec in csv.reader(io.StringIO(txt)):
        if not rec or rec[0].lstrip().startswith("#"):
            continue
        out.append((int(rec[0]), int(rec[1])))
    return out


def _edges_from_snap(path):
    with gzip.open(path, "rt") as f:
        return [tuple(int(x) for x in ln.split()[:2]) for ln in f if ln.strip() and not ln.startswith("#")]


def load_real(name):
    key = NAMES.get(name, name)
    cache = DATA / f"{key}.edges"
    if not cache.exists():
        _publish_cache(key, cache)
    G = nx.read_edgelist(cache, nodetype=int)
    return nx.convert_node_labels_to_integers(G, ordering="sorted")


def _publish_cache(key, cache):
    if key == "email-Eu-core":
        edges = _edges_from_snap(DATA / "email-Eu-core.txt.gz")
    else:
        edges = _edges_from_zip(DATA / f"{key}.zip")
    G = nx.Graph(); G.add_edges_from((u, v) for u, v in edges if u != v)
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    import os, uuid
    tmp = cache.with_name(f".{cache.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"); nx.write_edgelist(G, tmp, data=False); os.replace(tmp, cache)


def real_target(name, shift=None):
    G = load_real(name); t = PT.target_from(G)
    if t is None:
        raise RuntimeError(f"{name}: non-finite metric")
    for k, d in (shift or {}).items():
        t[k] = round(float(t[k]) + float(d), 6)
    t["_n"] = G.number_of_nodes(); t["_kbar"] = round(2.0 * G.number_of_edges() / G.number_of_nodes(), 4)
    t["_graph_edges"] = [(int(u), int(v)) for u, v in G.edges()]
    t["_degseq"] = sorted((d for _, d in G.degree()), reverse=True)
    t["name"] = name
    return t


def parse_real_source(src):
    assert src.startswith("real:"), src
    out = []
    for spec in src[5:].split(";"):
        spec = spec.strip()
        if not spec:
            continue
        name, _, sh = spec.partition("@")
        shift = {}
        if sh:
            for kv in sh.split(","):
                k, v = kv.split("="); shift[k.strip()] = float(v)
        label = name + ("".join(f"@{k}{v:+.3f}" for k, v in shift.items()) if shift else "")
        out.append((real_target(name, shift), label))
    return out


_RF = None


def _train_family_rf(n_per=60, seeds=(50000, 50001, 50002)):
    from sklearn.ensemble import RandomForestClassifier
    X, y = [], []
    for s in seeds:
        for d, f in PT.sample_targets(n_per * len(PT.FAMILIES) // len(seeds) // 1, s):
            X.append([d[k] for k in KEYS]); y.append(f)
    rf = RandomForestClassifier(300, random_state=0).fit(np.asarray(X, float), np.asarray(y)); return rf


def estimate_family(t):
    global _RF
    if _RF is None:
        cache = DATA / "family_rf.pkl"
        try:
            import pickle
            _RF = pickle.load(open(cache, "rb"))
        except Exception:
            _RF = _train_family_rf()
            import pickle; pickle.dump(_RF, open(cache, "wb"))
    x = np.asarray([[t[k] for k in KEYS]], float)
    probs = dict(zip(_RF.classes_, _RF.predict_proba(x)[0]))
    return max(probs, key=probs.get), probs


def _gen(family, n, kbar, rng):
    m = max(1, int(round(kbar / 2))); s = int(rng.integers(1 << 30))
    if family == "ws":
        return nx.watts_strogatz_graph(n, max(2, int(round(kbar))), float(rng.uniform(0.02, 0.6)), seed=s)
    if family == "holme_kim":
        return nx.powerlaw_cluster_graph(n, m, float(rng.uniform(0.0, 1.0)), seed=s)
    if family == "extended_ba":
        return nx.extended_barabasi_albert_graph(n, m, float(rng.uniform(0.0, 0.15)), float(rng.uniform(0.0, 0.15)), seed=s)
    if family == "sbm":
        b = int(rng.integers(2, 7)); sizes = [n // b] * b; sizes[0] += n - sum(sizes); sz = n / b
        ratio = float(rng.uniform(1.5, 20.0)); pout = kbar / (ratio * (sz - 1) + (n - sz)); pin = min(1.0, ratio * pout)
        P = np.full((b, b), pout); np.fill_diagonal(P, pin)
        return nx.stochastic_block_model(sizes, P, seed=s)
    if family == "soft_rgg":
        r = float(np.sqrt(kbar / (n * np.pi)) * rng.uniform(0.95, 1.25))
        return nx.soft_random_geometric_graph(n, r, seed=s)
    if family == "gaussian_part":
        sz = int(rng.integers(max(5, n // 10), max(6, n // 3))); v = float(rng.uniform(5, 20))
        pin = float(rng.uniform(0.05, 0.16)) * (200.0 / n) ** 0.0; pout = kbar / n * float(rng.uniform(0.3, 1.0))
        return nx.gaussian_random_partition_graph(n, sz, v, min(1.0, pin * kbar / 6.0), min(1.0, pout), seed=s)
    if family == "relaxed_caveman":
        c = max(3, int(round(kbar)) + 1); k = max(2, n // c)
        return nx.relaxed_caveman_graph(k, c, float(rng.uniform(0.1, 0.5)), seed=s)
    raise ValueError(family)


def _adjust_density(G, kbar, rng, tol=0.02):
    n = G.number_of_nodes(); target_m = int(round(kbar * n / 2)); rr = random.Random(int(rng.integers(1 << 30)))
    nodes = list(G.nodes()); guard = 0
    while abs(G.number_of_edges() - target_m) > tol * target_m and guard < 50 * n:
        guard += 1
        if G.number_of_edges() < target_m:
            u, v = rr.sample(nodes, 2)
            if not G.has_edge(u, v):
                G.add_edge(u, v)
        else:
            u, v = rr.choice(list(G.edges()))
            if G.degree(u) > 1 and G.degree(v) > 1:
                G.remove_edge(u, v)
                if not nx.has_path(G, u, v):
                    G.add_edge(u, v)
    return G


def family_graph(family, n, kbar, rng, tol=0.08, tries=200):
    best = None
    for _ in range(tries):
        try:
            G = _gen(family, n, kbar, rng)
        except Exception:
            continue
        G = nx.Graph(G); G.remove_edges_from(nx.selfloop_edges(G))
        if G.number_of_nodes() != n:
            G.add_nodes_from(range(n))
        G = nx.convert_node_labels_to_integers(G) if set(G.nodes()) != set(range(n)) else G
        kb = 2 * G.number_of_edges() / n
        if abs(kb - kbar) > tol * kbar:
            if best is None or abs(kb - kbar) < best[0]:
                best = (abs(kb - kbar), G)
            continue
        return _adjust_density(_connect(G, rng), kbar, rng)
    if best is None:
        raise RuntimeError(f"family_graph({family}, n={n}, kbar={kbar}): no graph")
    return _adjust_density(_connect(best[1], rng), kbar, rng)


def _connect(G, rng):
    if nx.is_connected(G):
        return G
    comps = sorted(nx.connected_components(G), key=len, reverse=True); big = set(comps[0])
    rr = random.Random(int(rng.integers(1 << 30)))
    for comp in comps[1:]:
        u = rr.choice(sorted(comp)); v = rr.choice(sorted(big)); G.add_edge(u, v); big |= comp
        for (a, b) in rr.sample(list(G.edges()), min(200, G.number_of_edges())):
            if a in big and b in big and (a, b) != (u, v) and (b, a) != (u, v):
                G.remove_edge(a, b)
                if nx.is_connected(G.subgraph(big)):
                    break
                G.add_edge(a, b)
    return G


class FamilyEnv(base.SwapEnv):

    def __init__(self, spec, t, family=None):
        super().__init__(spec, forward_sweep=False)
        self.n0 = int(t["_n"]); self.kbar0 = float(t["_kbar"])
        self.family, self.probs = (family, None) if family else estimate_family(t)
        self.info = {}

    def _start(self, seed):
        if seed in self._cache:
            return self._cache[seed].copy()
        G = family_graph(self.family, self.n0, self.kbar0, np.random.default_rng(7919 * seed + 29))
        self.info[seed] = dict(fam_est=self.family, fam_prob=round(float(self.probs[self.family]), 3) if self.probs else np.nan,
                               kbar_start=round(2 * G.number_of_edges() / self.n0, 3))
        self._cache[seed] = G.copy()
        return G
