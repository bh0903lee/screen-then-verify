"""Edit-conditioned graph neural rankers (GIN, GIN with positional encodings, GPS)."""
import hashlib, pickle
from pathlib import Path
import numpy as np
import networkx as nx

N_PE = 8; N_RW = 8; N_CHEAP = 4
FAMILIES = ("gin", "ginpe", "gps")


def node_inputs(n, edges, pe=True):
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from((int(u), int(v)) for u, v in edges)
    deg = np.array([G.degree(v) for v in range(n)], np.float64)
    clu = nx.clustering(G); tri = nx.triangles(G)
    nbr = np.array([np.mean([deg[w] for w in G.neighbors(v)]) if deg[v] > 0 else 0.0 for v in range(n)])
    cheap = np.stack([deg, [clu[v] for v in range(n)], [tri[v] for v in range(n)], nbr], 1)
    cheap = cheap / np.maximum(cheap.max(0, keepdims=True), 1e-9)
    cols = [cheap]
    if pe:
        A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float64); d = A.sum(1); dis = np.where(d > 0, 1.0 / np.sqrt(np.maximum(d, 1e-12)), 0.0)
        L = np.eye(n) - dis[:, None] * A * dis[None, :]
        if n > 1500:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            vals, vecs = eigsh(csr_matrix(L), k=N_PE + 1, sigma=-1e-6, which="LM"); o = np.argsort(vals); vecs = vecs[:, o]
        else:
            vals, vecs = np.linalg.eigh(L)
        pe_ = np.zeros((n, N_PE)); k = min(N_PE, vecs.shape[1] - 1); pe_[:, :k] = vecs[:, 1:1 + k]
        Pm = A / np.maximum(d, 1e-12)[:, None]; rw = np.zeros((n, N_RW)); Pk = np.eye(n)
        if n <= 1500:
            for t in range(N_RW):
                Pk = Pk @ Pm; rw[:, t] = np.diag(Pk)
        cols += [pe_, rw]
    X = np.concatenate(cols, 1).astype(np.float32)
    E = np.asarray(list(G.edges()), np.int64).reshape(-1, 2)
    ei = np.concatenate([E, E[:, ::-1]], 0).T if len(E) else np.zeros((2, 0), np.int64)
    return X, ei


def op_edges(op):
    import operators as P
    rem, add = P.edge_lists(op)
    return [(int(u), int(v)) for u, v in rem], [(int(u), int(v)) for u, v in add]


def build(family, n_in, n_local, d=64, layers=4, heads=4, hidden=128):
    import torch, torch.nn as nn

    class GIN(nn.Module):
        def __init__(self, din, dout):
            super().__init__(); self.mlp = nn.Sequential(nn.Linear(din, dout), nn.ReLU(), nn.Linear(dout, dout)); self.eps = nn.Parameter(torch.zeros(1))

        def forward(self, h, ei):
            agg = torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]])
            return self.mlp((1 + self.eps) * h + agg)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.family = family; self.inp = nn.Linear(n_in, d)
            self.gin = nn.ModuleList([GIN(d, d) for _ in range(layers)]); self.norm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
            self.att = nn.ModuleList([nn.MultiheadAttention(d, heads, batch_first=True) for _ in range(layers)]) if family == "gps" else None
            self.anorm = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]) if family == "gps" else None
            self.head = nn.Sequential(nn.Linear(2 * d + d + n_local, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def encode(self, X, ei, batch, nb):
            h = torch.relu(self.inp(X))
            for i in range(len(self.gin)):
                h = self.norm[i](h + torch.relu(self.gin[i](h, ei)))
                if self.att is not None:
                    counts = torch.bincount(batch, minlength=nb); m = int(counts.max())
                    pos = torch.arange(len(batch), device=h.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
                    Hd = h.new_zeros(nb, m, h.shape[1]); Hd[batch, pos] = h
                    mask = torch.ones(nb, m, dtype=torch.bool, device=h.device); mask[batch, pos] = False
                    a, _ = self.att[i](Hd, Hd, Hd, key_padding_mask=mask, need_weights=False)
                    h = self.anorm[i](h + a[batch, pos])
            g = torch.zeros(nb, h.shape[1], device=h.device).index_add_(0, batch, h) / torch.bincount(batch, minlength=nb).clamp(min=1)[:, None].float()
            return h, g

        def score(self, h, g, cand_graph, add_e, add_c, rem_e, rem_c, F):
            def phi(e):
                hu, hv = h[e[:, 0]], h[e[:, 1]]
                return torch.cat([hu * hv, (hu - hv).abs()], 1)
            C = cand_graph.shape[0]; e = h.new_zeros(C, 2 * h.shape[1])
            if len(add_e):
                e = e.index_add(0, add_c, phi(add_e))
            if len(rem_e):
                e = e.index_add(0, rem_c, -phi(rem_e))
            return self.head(torch.cat([e, g[cand_graph], F], 1)).squeeze(1)

    return Net()


def collate(states, device, pe):
    import torch
    Xs, eis, bat, cg, ae, ac, re_, rc, Fs = [], [], [], [], [], [], [], [], []
    off = 0; coff = 0
    for gi, s in enumerate(states):
        X, ei = s["X"], s["ei"]; n = X.shape[0]
        Xs.append(X); eis.append(ei + off); bat.append(np.full(n, gi, np.int64))
        for ci, op in enumerate(s["ops"]):
            rem, add = op_edges(op)
            for u, v in add:
                ae.append((u + off, v + off)); ac.append(coff + ci)
            for u, v in rem:
                re_.append((u + off, v + off)); rc.append(coff + ci)
        cg.append(np.full(len(s["ops"]), gi, np.int64)); Fs.append(s["F"]); off += n; coff += len(s["ops"])
    T = lambda a, dt=torch.long: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
    return dict(X=T(np.concatenate(Xs), torch.float32), ei=T(np.concatenate(eis, 1)), batch=T(np.concatenate(bat)), nb=len(states),
                cg=T(np.concatenate(cg)), ae=T(np.asarray(ae, np.int64).reshape(-1, 2)), ac=T(np.asarray(ac, np.int64)),
                re=T(np.asarray(re_, np.int64).reshape(-1, 2)), rc=T(np.asarray(rc, np.int64)), F=T(np.concatenate(Fs), torch.float32))


def forward(net, b):
    h, g = net.encode(b["X"], b["ei"], b["batch"], b["nb"])
    return net.score(h, g, b["cg"], b["ae"], b["ac"], b["re"], b["rc"], b["F"])


def _w0(r0):
    return float(np.abs(np.asarray(r0, np.float64)).max())


class NNPred:
    def __init__(self, d, path):
        import torch
        self.meta = d["meta"]; self.path = path; self.torch = torch
        self.net = build(self.meta["family"], int(self.meta["n_in"]), int(self.meta["n_local"]), **self.meta.get("arch", {}))
        self.net.load_state_dict(d["state"]); self.net.eval()
        self.mu = np.asarray(self.meta["x_mu"], np.float32); self.sd = np.asarray(self.meta["x_sd"], np.float32)
        self.hash = hashlib.md5(Path(path).read_bytes()).hexdigest()[:12]

    def score(self, G, ops, F_local, r0, rh=None, What=None):
        import predictors as PR
        n = G.number_of_nodes(); X, ei = node_inputs(n, G.edges(), pe=self.meta["family"] != "gin")
        if self.meta.get("feat", "L") == "LS":
            assert rh is not None and What is not None, "LS model needs the sketch features"
            Fx = PR.build_X(F_local, r0, _w0(r0), rh, What)
        else:
            Fx = PR.build_X(F_local, r0, 0.0)
        F = (Fx - self.mu) / self.sd
        b = collate([dict(X=X, ei=ei, ops=ops, F=F.astype(np.float32))], "cpu", None)
        with self.torch.no_grad():
            return forward(self.net, b).numpy().astype(np.float64)


def load(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"nnpred model {p} missing (experiments/train_ranker.py)")
    return NNPred(pickle.loads(p.read_bytes()), p)
