"""Shared helpers for targets, bands and residuals."""
import os
import sys
import logging
from dataclasses import dataclass
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

import graph_env as base
import ws_targets as e37

logger = logging.getLogger("common")
_dev_env = os.environ.get("D0_DEVICE", "").strip()
if _dev_env:
    DEV = torch.device(_dev_env)
elif torch.cuda.is_available():
    DEV = torch.device("cuda")
else:
    DEV = torch.device("cpu")
KEYS = base.KEYS
RESULTS_DIR = _REPO / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class D0Config:
    encoder: str = "gcn"
    pe: str = "fiedler"
    k_pe: int = 6
    emb: int = 32
    lookahead: bool = False
    preference: bool = False
    rl: str = "reinforce"
    frac_bridge: float = 0.5
    gamma: float = 0.97
    lr: float = 1e-3
    T: int = base.T_SWAPS
    K: int = base.K_CANDIDATES
    ppo_epochs: int = 4
    ppo_clip: float = 0.2

    def cand_cfg(self):
        return base.Ablation("d0", k_pe=self.k_pe, frac_bridge=self.frac_bridge)


def dense_raw_adj(G, n):
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32)
    return torch.tensor(A, device=DEV)


def norm_adj_dev(G, n):
    import scipy.sparse as sp
    A = nx.to_scipy_sparse_array(G, nodelist=range(n), format="csr", dtype=np.float32)
    A = A + sp.eye(n, dtype=np.float32)
    d = np.asarray(A.sum(axis=1)).flatten()
    Dinv = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-8)))
    return torch.tensor((Dinv @ A @ Dinv).toarray(), device=DEV)


def rwpe(G, n, k):
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float64)
    d = A.sum(1); d[d == 0] = 1.0
    RW = A / d[:, None]
    P = np.eye(n); out = np.zeros((n, k), np.float32)
    for t in range(k):
        P = P @ RW
        out[:, t] = np.diag(P).astype(np.float32)
    return out


def node_features(G, n, cfg, pe_cache=None, sign_flip=False):
    if cfg.pe == "fiedler":
        return base.node_features_pe(G, n, cfg.k_pe, sign_flip=sign_flip, pe=pe_cache)
    base_feat = base.cheap_node_features(G)
    if cfg.pe == "none" or cfg.k_pe <= 0:
        return base_feat
    if cfg.pe == "rwpe":
        return np.concatenate([base_feat, rwpe(G, n, cfg.k_pe)], axis=1)
    raise ValueError(cfg.pe)


class Encoder(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.cfg = cfg; e = cfg.emb
        if cfg.encoder == "gcn":
            self.w1 = nn.Linear(in_dim, e); self.w2 = nn.Linear(e, e)
        elif cfg.encoder == "gin":
            self.eps1 = nn.Parameter(torch.zeros(1)); self.eps2 = nn.Parameter(torch.zeros(1))
            self.m1 = nn.Sequential(nn.Linear(in_dim, e), nn.ReLU(), nn.Linear(e, e))
            self.m2 = nn.Sequential(nn.Linear(e, e), nn.ReLU(), nn.Linear(e, e))
        elif cfg.encoder == "gat":
            self.lin1 = nn.Linear(in_dim, e); self.att1 = nn.Linear(2 * e, 1)
            self.lin2 = nn.Linear(e, e); self.att2 = nn.Linear(2 * e, 1)
        else:
            raise ValueError(cfg.encoder)

    def _gat_layer(self, A, X, lin, att):
        n = X.shape[0]
        h = lin(X)
        hi = h.unsqueeze(1).expand(n, n, -1)
        hj = h.unsqueeze(0).expand(n, n, -1)
        e = F.leaky_relu(att(torch.cat([hi, hj], -1)).squeeze(-1), 0.2)
        e = e.masked_fill(A <= 0, float("-inf"))
        alpha = torch.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha)
        return F.relu(alpha @ h)

    def forward(self, An, Araw, X):
        c = self.cfg.encoder
        if c == "gcn":
            h = F.relu(self.w1(An @ X)); h = F.relu(self.w2(An @ h)); return h
        if c == "gin":
            A = Araw
            h = self.m1((1 + self.eps1) * X + A @ X); h = F.relu(h)
            h = self.m2((1 + self.eps2) * h + A @ h); return F.relu(h)
        if c == "gat":
            h = self._gat_layer(Araw, X, self.lin1, self.att1)
            h = self._gat_layer(Araw, h, self.lin2, self.att2)
            return h


class D0Policy(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(in_dim, cfg)
        e = cfg.emb
        extra = (2 if cfg.lookahead else 0) + (5 if cfg.preference else 0)
        self.scorer = nn.Sequential(nn.Linear(4 * e + 5 + extra, 64), nn.ReLU(), nn.Linear(64, 1))
        vin = e + 5 + (5 if cfg.preference else 0)
        self.value = nn.Sequential(nn.Linear(vin, 64), nn.ReLU(), nn.Linear(64, 1))

    def encode(self, An, Araw, X):
        return self.enc(An, Araw, X)

    def score(self, h, cand, res, cand_feat=None, w=None):
        emb = h[cand].reshape(cand.shape[0], -1)
        r = res.unsqueeze(0).expand(emb.shape[0], -1)
        parts = [emb, r]
        if cand_feat is not None:
            parts.append(cand_feat)
        if w is not None:
            parts.append(w.unsqueeze(0).expand(emb.shape[0], -1))
        return self.scorer(torch.cat(parts, dim=1)).squeeze(1)

    def baseline(self, h, res, w=None):
        parts = [h.mean(dim=0), res]
        if w is not None:
            parts.append(w)
        return self.value(torch.cat(parts, dim=0)).squeeze()


sample_targets = e37.sample_targets
eval_fixed = base.eval_fixed


