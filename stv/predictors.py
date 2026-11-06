"""Features and models of the direct learned predictors."""
import hashlib, pickle
from pathlib import Path
import numpy as np

FEATS = ("L", "LS")
GFEAT = "G"


def build_X(F_local, r0, W0, rh=None, What=None):
    F = np.asarray(F_local, np.float32); n = len(F); r0 = np.asarray(r0, np.float32).reshape(1, -1)
    cols = [F, np.repeat(r0, n, 0), np.repeat(np.abs(r0), n, 0)]
    if rh is not None:
        rh = np.asarray(rh, np.float32).reshape(n, -1); What = np.asarray(What, np.float32).reshape(n, 1)
        cols += [rh, What, What - np.float32(W0)]
    return np.hstack(cols).astype(np.float32)


def build_X_table(F, df, feat):
    r0 = df[[c for c in df.columns if c.startswith("r0_")]].values.astype(np.float32)
    cols = [np.asarray(F, np.float32), r0, np.abs(r0)]
    if feat == "LS":
        rh = df[[c for c in df.columns if c.startswith("rh_")]].values.astype(np.float32)
        cols += [rh, df[["What"]].values.astype(np.float32), (df.What - df.W0).values.astype(np.float32).reshape(-1, 1)]
    return np.hstack(cols).astype(np.float32)


class Pred:
    def __init__(self, model, feat, arm, meta, path=None):
        self.model, self.feat, self.arm, self.meta, self.path = model, feat, arm, meta, path
        self.hash = hashlib.md5(Path(path).read_bytes()).hexdigest()[:12] if path else ""

    def predict(self, X):
        return np.asarray(self.model.predict(np.asarray(X, np.float32)), np.float64)


OBJ_OF = {"pred": "top1", "preds": "top1", "pred_k8": "top8", "preds_k8": "top8", "gpred": "top1", "gpred_k8": "top8"}
FEAT_OF = {"pred": "L", "pred_k8": "L", "preds": "LS", "preds_k8": "LS", "gpred": "G", "gpred_k8": "G"}


def model_path(pred_dir, arm, feat, obj="top8"):
    return Path(pred_dir) / f"model_{arm}_{feat}_{obj}.pkl"


def model_digest(pred_dir, arm, policy):
    f = model_path(pred_dir, arm, FEAT_OF[policy], OBJ_OF[policy])
    return hashlib.md5(f.read_bytes()).hexdigest()[:12] if f.exists() else "missing"


def model_digest_lofo(pred_dir, arm, policy):
    import targets as PT
    return ";".join(f"{f}:{model_digest(Path(pred_dir) / f'excl_{f}', arm, policy)}" for f in PT.FAMILIES)


def load(pred_dir, arm, feat, obj="top8"):
    if not pred_dir:
        raise ValueError("pred / preds policies need --pred-dir (models written by train_direct_predictor.py)")
    f = model_path(pred_dir, arm, feat, obj)
    if not f.exists():
        raise FileNotFoundError(f"direct-predictor model {f} missing (run train_direct_predictor.py --arm {arm})")
    d = pickle.loads(f.read_bytes())
    assert d["feat"] == feat and d["arm"] == arm, f"model {f} is for ({d['arm']}, {d['feat']})"
    if feat == GFEAT:
        return GPred(d["state"], arm, d["meta"], path=f)
    return Pred(d["model"], feat, arm, d.get("meta", {}), path=f)


N_NODE_FEAT = 4


def gnet_adj(G, n):
    import networkx as nx, torch
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32) + np.eye(n, dtype=np.float32)
    d = 1.0 / np.sqrt(A.sum(1)); An = (A * d[:, None]) * d[None, :]
    return torch.from_numpy(np.ascontiguousarray(An))


def gnet_tensors(G, n):
    import torch, graph_env as base
    return gnet_adj(G, n), torch.from_numpy(np.asarray(base.cheap_node_features(G), np.float32))


def local_vec(F_local, r0):
    return build_X(F_local, r0, 0.0)


def make_gnet(n_local, emb=64, hidden=64):
    import torch.nn as nn

    class GNet(nn.Module):
        def __init__(self):
            super().__init__(); self.w1 = nn.Linear(N_NODE_FEAT, emb); self.w2 = nn.Linear(emb, emb)
            self.head = nn.Sequential(nn.Linear(emb + n_local, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, An, X, f):
            import torch
            h = torch.relu(self.w1(An @ X)); h = torch.relu(self.w2(An @ h)); g = h.mean(0)
            return self.head(torch.cat([g, f], 0)).squeeze()
    return GNet()


class GPred:
    def __init__(self, state, arm, meta, path=None):
        import torch, operators as P
        self.arm, self.meta, self.path, self._P, self._torch = arm, meta, path, P, torch
        self.net = make_gnet(int(meta["n_local"]), int(meta["emb"]), int(meta["hidden"])); self.net.load_state_dict(state); self.net.eval()
        self.mu = np.asarray(meta["x_mu"], np.float32); self.sd = np.asarray(meta["x_sd"], np.float32)
        self.hash = hashlib.md5(Path(path).read_bytes()).hexdigest()[:12] if path else ""

    def predict_cands(self, G, ops, F_local, r0):
        torch, P = self._torch, self._P; n = G.number_of_nodes()
        Xl = (local_vec(F_local, r0) - self.mu) / self.sd; out = np.zeros(len(ops), np.float64)
        with torch.no_grad():
            for i, op in enumerate(ops):
                P.apply_op(G, op)
                try:
                    An, X = gnet_tensors(G, n); out[i] = float(self.net(An, X, torch.from_numpy(np.asarray(Xl[i], np.float32))))
                finally:
                    P.revert_op(G, op)
        return out
