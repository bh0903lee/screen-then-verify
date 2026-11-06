"""Exact metric oracle (clustering, assortativity, average path length, modularity, spectral gap), tolerance bands and the edit environment."""
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from warmstart import WarmStartGenerator, ConstraintSpec
from warmstart.metrics import compute_metrics
from warmstart.stages import (
    Stage1DegreeSequence, Stage2Clustering, Stage3PathLength,
    Stage4Community, Stage5SpectralGap, Stage6Assortativity,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("graph_env")
logging.getLogger("warmstart").setLevel(logging.WARNING)

RESULTS_DIR = _REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)

N_NODES = 200
AVG_DEG = 6.0
DEGREE_SEQ = None
KEYS = ["clustering", "avg_path_length", "modularity", "spectral_gap", "assortativity"]
ACTIVE = np.ones(5, np.float32)
TOL = {"clustering": 0.05, "avg_path_length": 0.30, "modularity": 0.05,
       "spectral_gap": 0.03, "assortativity": 0.05}
SCENARIOS = {
    "A": dict(clustering=0.25, avg_path_length=3.70, modularity=0.55, spectral_gap=0.15, assortativity=-0.10),
    "B": dict(clustering=0.25, avg_path_length=3.70, modularity=0.55, spectral_gap=0.12, assortativity=-0.20),
    "C": dict(clustering=0.35, avg_path_length=3.80, modularity=0.60, spectral_gap=0.09, assortativity=+0.15),
    "D": dict(clustering=0.20, avg_path_length=3.40, modularity=0.52, spectral_gap=0.16, assortativity=+0.05),
    "E": dict(clustering=0.28, avg_path_length=3.70, modularity=0.57, spectral_gap=0.10, assortativity=-0.18),
}
STAGE_CLASSES = [Stage2Clustering, Stage3PathLength, Stage4Community,
                 Stage5SpectralGap, Stage6Assortativity]
STAGE_CFG = {"stage2": {"max_iterations": 30_000}, "stage5": {"max_iterations": 3_000}}

K_CANDIDATES = 32
T_SWAPS = 60
K_PE_DEFAULT = 6
FRAC_BRIDGE_DEFAULT = 0.5
EMB_DIM = 32
SOLVE_BONUS = 5.0
STEP_COST = 0.02
DEVICE = torch.device("cpu")
W_POT = np.array([1.0, 1.0, 1.0, 3.0, 1.0], dtype=np.float32)


@dataclass
class Ablation:
    name: str
    k_pe: int = K_PE_DEFAULT
    frac_bridge: float = FRAC_BRIDGE_DEFAULT
    learned: bool = True
    forward_sweep: bool = True


VARIANTS = [
    Ablation("full"),
    Ablation("no_fiedler_pe", k_pe=0),
    Ablation("no_bridge", frac_bridge=0.0),
    Ablation("random_policy", learned=False),
    Ablation("from_scratch", forward_sweep=False),
]


def phi(res):
    return -float(np.average(np.abs(res), weights=W_POT))


def make_spec(params, n_nodes=None, avg_degree=None):
    p = dict(params)
    dseq = p.pop("degree_seq", None)
    if dseq is None:
        dseq = DEGREE_SEQ
    return ConstraintSpec(n_nodes=n_nodes if n_nodes is not None else N_NODES,
                          avg_degree=avg_degree if avg_degree is not None else AVG_DEG,
                          tolerances=TOL, degree_seq=dseq, **p)


def cheap_node_features(G):
    nodes = list(range(G.number_of_nodes()))
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    clu = nx.clustering(G); clu = np.array([clu[n] for n in nodes], dtype=np.float32)
    tri = nx.triangles(G); tri = np.array([tri[n] for n in nodes], dtype=np.float32)
    and_ = np.zeros(len(nodes), dtype=np.float32)
    for i, n in enumerate(nodes):
        nbrs = list(G.neighbors(n))
        if nbrs:
            and_[i] = np.mean([G.degree(nb) for nb in nbrs])
    return np.stack([deg / max(deg.max(), 1.0), clu,
                     tri / max(tri.max(), 1.0), and_ / max(and_.max(), 1.0)], axis=1)


def laplacian_pe(G, n, k):
    if k <= 0 or G.number_of_edges() == 0:
        return np.zeros((n, max(k, 0)), dtype=np.float32)
    try:
        L = nx.normalized_laplacian_matrix(G, nodelist=range(n)).toarray().astype(np.float64)
        _, vecs = np.linalg.eigh(L)
        spec = vecs[:, 1:1 + min(k, n - 1)].astype(np.float32)
        if spec.shape[1] < k:
            spec = np.concatenate([spec, np.zeros((n, k - spec.shape[1]), np.float32)], axis=1)
        return spec
    except Exception:
        return np.zeros((n, k), dtype=np.float32)


def node_features_pe(G, n, k_pe, sign_flip=False, pe=None):
    base = cheap_node_features(G)
    if k_pe <= 0:
        return base
    if pe is None:
        pe = laplacian_pe(G, n, k_pe)
    if sign_flip and pe.shape[1] > 0:
        signs = np.random.choice([-1.0, 1.0], size=pe.shape[1]).astype(np.float32)
        pe = pe * signs[None, :]
    return np.concatenate([base, pe], axis=1)


def sample_candidates(G, k):
    edges = list(G.edges()); cands = []; tries = 0
    while len(cands) < k and tries < k * 8:
        tries += 1
        (a, b), (c, d) = random.choice(edges), random.choice(edges)
        if len({a, b, c, d}) < 4:
            continue
        if G.has_edge(a, d) or G.has_edge(c, b):
            continue
        cands.append((a, b, c, d))
    return cands


def sample_candidates_spectral(G, k, fiedler, frac_bridge):
    if frac_bridge <= 0 or fiedler is None or fiedler.size == 0:
        return sample_candidates(G, k)
    pos = fiedler >= 0
    edges = list(G.edges())
    pos_edges = [(u, v) for u, v in edges if pos[u] and pos[v]]
    neg_edges = [(u, v) for u, v in edges if (not pos[u]) and (not pos[v])]
    cands, seen = [], set()
    n_bridge = int(k * frac_bridge); tries = 0
    while len(cands) < n_bridge and pos_edges and neg_edges and tries < k * 10:
        tries += 1
        a, b = random.choice(pos_edges); c, d = random.choice(neg_edges)
        if len({a, b, c, d}) < 4 or G.has_edge(a, d) or G.has_edge(c, b):
            continue
        key = (a, b, c, d)
        if key in seen:
            continue
        seen.add(key); cands.append(key)
    cands += sample_candidates(G, k - len(cands))
    return cands[:k]


class SwapPolicy(nn.Module):
    def __init__(self, in_dim, emb=EMB_DIM, res_dim=5):
        super().__init__()
        self.w1 = nn.Linear(in_dim, emb); self.w2 = nn.Linear(emb, emb)
        self.scorer = nn.Sequential(nn.Linear(4 * emb + res_dim, 64), nn.ReLU(), nn.Linear(64, 1))
        self.value = nn.Sequential(nn.Linear(emb + res_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, An, X):
        h = F.relu(self.w1(An @ X)); h = F.relu(self.w2(An @ h)); return h

    def score(self, h, cand, res):
        e = h[cand].reshape(cand.shape[0], -1)
        r = res.unsqueeze(0).expand(e.shape[0], -1)
        return self.scorer(torch.cat([e, r], dim=1)).squeeze(1)

    def baseline(self, h, res):
        return self.value(torch.cat([h.mean(dim=0), res], dim=0)).squeeze()


class SwapEnv:
    def __init__(self, spec, forward_sweep=True):
        self.spec = spec; self.targets = spec.scalar_targets
        self.forward_sweep = forward_sweep; self._cache = {}

    def _start(self, seed):
        if seed in self._cache:
            return self._cache[seed].copy()
        random.seed(seed); np.random.seed(seed)
        G = Stage1DegreeSequence(self.spec, verbose=False).run()
        if self.forward_sweep:
            _skey = {Stage2Clustering: "stage2", Stage3PathLength: "stage3", Stage4Community: "stage4",
                     Stage5SpectralGap: "stage5", Stage6Assortativity: "stage6"}
            for StageCls in STAGE_CLASSES:
                G = StageCls(self.spec, **STAGE_CFG.get(_skey.get(StageCls, ""), {}), verbose=False).run(G)
        self._cache[seed] = G.copy(); return G

    def residuals(self, G):
        m = compute_metrics(G, KEYS)
        r = np.array([(m[k] - self.targets[k]) / TOL[k] for k in KEYS], np.float32)
        if not ACTIVE.all():
            r = r * ACTIVE
        return r

    def reset(self, seed):
        self.G = self._start(seed); self.res = self.residuals(self.G)
        return self.G, self.res


def eval_fixed(spec, n_trials, seed0=10_000):
    rows = []
    for t in range(n_trials):
        random.seed(seed0 + t); np.random.seed(seed0 + t); t0 = time.time()
        G = WarmStartGenerator(spec, n_passes=5, verbose=False, stage_configs=STAGE_CFG).generate()
        m = compute_metrics(G, KEYS)
        res = np.array([(m[k] - spec.scalar_targets[k]) / TOL[k] for k in KEYS])
        rows.append({"trial": t, "elapsed": round(time.time() - t0, 2),
                     "overall_sat": int(np.all(np.abs(res) <= 1.0)),
                     "n_sat": int(np.sum(np.abs(res) <= 1.0))})
    return pd.DataFrame(rows)


