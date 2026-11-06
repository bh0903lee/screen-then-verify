"""One run = (operator, seed, selectors) on the evaluation targets; writes one CSV per selector."""
import argparse
import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd, networkx as nx, torch, torch.nn as nn

import graph_env as base, ws_targets as e37, common as d0
import bands as TB, io_utils, operators as P, targets as PT, sketch as SK, predictors as PR
import result_check as RC
import real_networks as REAL
import extensions as EXT

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logging.getLogger("warmstart").setLevel(logging.WARNING); log = logging.getLogger("run")
DEV = d0.DEV
EXP = "run"
BANDS = ["u1.0", "u0.8", "u0.6", "u0.5", "u0.45", "u0.4", "u0.35", "u0.3", "u0.25", "u0.2", "u0.15", "u0.1"]
TRAIN_BAND = "u0.3"
POLICIES = ["random", "greedy", "value", "value_rel", "value_af", "localfit"]
LABEL_OF = {"value": None, "value_af": None, "value_rel": "relative",
            "value_rel_res": "relative", "value_rel_const": "relative", "value_sketch": "relative",
            "value_sketch_min": "relative", "value_h10": "relative", "value_h30": "relative", "value_td": "td", "value_mc": "td"}
VALUE_KINDS = ("value", "value_af", "value_rel", "value_rel_res", "value_sketch", "value_sketch_min",
               "value_h10", "value_h30", "value_td", "value_mc")
HORIZON_OF = {"value_h10": 10, "value_h30": 30}
TD_KINDS = ("value_td", "value_mc")
VALUE_POLICIES = VALUE_KINDS + ("value_rel_const",)
SKETCH_POLICIES = ("sketch", "sketch_k8")
PRED_POLICIES = ("pred", "pred_k8", "preds", "preds_k8", "gpred", "gpred_k8")
UCT_POLICIES = ("uct",)
SA_POLICIES = ("sa", "sa_free", "mcmc", "hill", "hill_free", "mcmcf02", "mcmcf05", "mcmcf10")
SA_UNCAPPED = {"sa_free": None, "hill_free": 0.0, "mcmcf02": 0.02, "mcmcf05": 0.05, "mcmcf10": 0.10}
STAGE1_ARMS = ["S_swap", "S_tswap", "S_jdm", "S_swap3", "S_rewire", "S_adddel", "S_tadddel"]
BASELINE_ARM = "S_swap"


def get_band(b):
    return TB.BAND_DEFS[b] if b in TB.BAND_DEFS else (np.full(5, float(b[1:]), np.float32), "worst")


def worst(res):
    return float(np.abs(np.asarray(res, np.float32)).max())


class Cfg:
    def __init__(self, **kw):
        self.T = 60; self.K = 32; self.greedy_k = 8; self.value_k = 8
        self.greedy_obj = "worst"
        self.train_targets = 14; self.eval_targets = 28; self.source = "mix7"
        self.depth = 3; self.b = 4; self.k_expand = 6; self.rollouts = 2; self.rollout_len = 10
        self.value_epochs = 1200; self.value_batch = 1; self.lr = 1e-3
        self.sig_states = 144; self.sig_k = 32
        self.kband = P.KBAND; self.mask_connected = True
        self.value_label = "clip3"; self.value_fam = False
        self.screen_mult = 4; self.beam_bands = ["u0.4", "u0.3"]
        self.sketch_k = 8; self.sketch_q_refresh = 1; self.need_sketch = False; self.sketch_lam_scale = 1.0
        self.warm = "ws"
        self.ckpt_tag = None
        self.noise_k = None; self.no_signal = False; self.pswap_model = None
        self.mixb_eps = 0.3; self.mixb_rollouts = 4; self.mixb_min_count = 3
        self.td_gamma = 0.97; self.td_bonus = 1.0; self.td_n = 5; self.td_eps = 0.2; self.td_iters = 3; self.td_rollouts = 2; self.td_steps = 3000
        self.oracle_lam = "eigsh"
        self.oracle_q = "cnm"
        self.nodes = 200
        self.uct_expand = 8; self.uct_depth = 3; self.uct_c = float(np.sqrt(2)); self.uct_widen = 0.0; self.uct_widen_c = 1.0; self.pred_dir = None; self.pred_hash = ""
        self.pred_lofo = False; self.lam_lofo = None
        self.sa_budget = None; self.sa_tau0 = 0.3; self.sa_tau1 = 0.01; self.sa_tau_fixed = 0.05
        self.save_graphs = False
        self.metrics = "full5"
        self.warm_cache = None
        self.arm_weights = None
        self.target_cache = None
        self.oracle_apl = "nx"; self.stop_band = None; self.ext_calib = None; self.ext_model = None
        self.max_oracle = None
        self.__dict__.update(kw)
        self.d0 = d0.D0Config(encoder="gcn", pe="none"); self.d0.T = self.T

    def as_dict(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items() if k != "d0"}


def is_screen(groups):
    sc = [g in P.SCREEN_GROUPS for g in groups]; dv = [g in P.DIVERSE_GROUPS for g in groups]
    if (any(sc) or any(dv)) and not (all(sc) or all(dv)):
        raise ValueError(f"oracle-screen groups cannot be mixed with other groups in one arm: {groups}")
    return "screen" if (groups and all(sc)) else ("diverse" if (groups and all(dv)) else "")


def pool_diversity(ops):
    sets = [P.op_nodes(op) for op in ops]
    if len(sets) < 2:
        return 0.0
    tot = 0.0; n = 0
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            u = len(sets[i] | sets[j]); tot += len(sets[i] & sets[j]) / u if u else 0.0; n += 1
    return tot / n


def candidates(G, groups, res, rng, cfg, with_fied=False, acc=None, env=None, weights=None, fam=None):
    if weights is None and getattr(cfg, "arm_weights", None) is not None:
        if set(cfg.arm_weights) != set(groups):
            raise ValueError(f"cfg.arm_weights {cfg.arm_weights} does not match groups {groups} (stale Cfg reused across arms?)")
        weights = cfg.arm_weights
    screen = is_screen(groups)
    K = cfg.K * cfg.screen_mult if screen else cfg.K
    pm = None
    ops, fied, st = P.gen_ops(G, groups, K, rng, res=res, kband=cfg.kband, mask_connected=cfg.mask_connected, weights=weights,
                              pswap_model=pm)
    if screen and ops:
        assert env is not None, "screen proposals need the oracle (env)"
        scored = sorted(((worst(eval_op(env, G, op)), i) for i, op in enumerate(ops)))
        if acc is not None:
            acc["screen_calls"] = acc.get("screen_calls", 0) + len(ops)
        if screen == "screen":
            sel = [ops[i] for _, i in scored[:cfg.K]]
        else:
            used = set(); sel = []; rest = []
            for _, i in scored:
                nd = P.op_nodes(ops[i])
                if nd & used:
                    rest.append(ops[i]); continue
                sel.append(ops[i]); used |= nd
                if len(sel) >= cfg.K:
                    break
            sel += rest[:cfg.K - len(sel)]
        if acc is not None:
            acc["div_sum"] = acc.get("div_sum", 0.0) + pool_diversity(sel); acc["div_n"] = acc.get("div_n", 0) + 1
        ops = sel
    if acc is not None:
        acc["steps"] = acc.get("steps", 0) + 1; acc["requested"] = acc.get("requested", 0) + cfg.K
        acc["produced"] = acc.get("produced", 0) + len(ops); acc["masked"] = acc.get("masked", 0) + st["masked"]
        acc["fallback"] = acc.get("fallback", 0) + sum(st["fallback"].values())
    return (ops, fied) if with_fied else ops


FAMILIES = ["ws", "holme_kim", "sbm", "soft_rgg", "gaussian_part", "relaxed_caveman", "extended_ba"]
FAM_DIM = len(FAMILIES)


def fam_onehot(fam):
    v = np.zeros(FAM_DIM, np.float32)
    if fam in FAMILIES:
        v[FAMILIES.index(fam)] = 1.0
    return v


def spec_of(t):
    metrics = {k: t[k] for k in base.KEYS}
    if "degree_seq" in t:
        seq = [int(x) for x in t["degree_seq"]]
        return base.make_spec({**metrics, "degree_seq": seq}, n_nodes=len(seq), avg_degree=float(sum(seq)) / len(seq))
    if "_n" in t and "_kbar" in t and t.get("name"):
        return base.make_spec(metrics, n_nodes=int(t["_n"]), avg_degree=float(t["_kbar"]))
    return base.make_spec(metrics)


def connect_graph(G, rng):
    reps = 0
    while not nx.is_connected(G):
        comps = sorted(nx.connected_components(G), key=len); small, big = comps[0], comps[-1]
        u = rng.choice(sorted(small)); w = rng.choice(sorted(big)); G.add_edge(u, w)
        big_edges = [(x, y) for x, y in G.edges() if x in big and y in big and (x, y) != (u, w) and (y, x) != (u, w)]
        rng.shuffle(big_edges); ncomp = nx.number_connected_components(G)
        for x, y in big_edges:
            if G.degree(x) > 1 and G.degree(y) > 1:
                G.remove_edge(x, y)
                if nx.number_connected_components(G) == ncomp:
                    break
                G.add_edge(x, y)
        reps += 1
        if reps > 200:
            break
    return G, reps


def connect_graph_degree_preserving(G, rng):
    reps = 0
    while not nx.is_connected(G):
        comps = sorted(nx.connected_components(G), key=len); S, L = comps[0], comps[-1]
        s_edges = sorted((min(x, y), max(x, y)) for x, y in G.subgraph(S).edges())
        bridges = set(map(lambda e: (min(e), max(e)), nx.bridges(G.subgraph(L))))
        l_edges = [e for e in sorted((min(x, y), max(x, y)) for x, y in G.subgraph(L).edges()) if e not in bridges]
        if not s_edges or not l_edges:
            raise RuntimeError("degree-preserving repair impossible (isolated node or tree-only largest component)")
        a, b = s_edges[rng.randrange(len(s_edges))]; c, d = l_edges[rng.randrange(len(l_edges))]
        G.remove_edge(a, b); G.remove_edge(c, d); G.add_edge(a, c); G.add_edge(b, d); reps += 1
        if reps > 1000:
            raise RuntimeError("degree-preserving repair did not converge")
    return G, reps


def graph_hash(G):
    e = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in G.edges())
    return hashlib.md5(json.dumps([G.number_of_nodes(), e]).encode()).hexdigest()[:12]


def reset_connected(env, sd):
    G, res = env.reset(sd)
    reps_log = env.__dict__.setdefault("_repairs", {})
    if nx.is_connected(G):
        return G, res, reps_log.get(sd, 0)
    dseq = getattr(getattr(env, "spec", None), "degree_seq", None)
    if dseq is not None:
        G2, reps = connect_graph_degree_preserving(G.copy(), random.Random(sd * 31 + 5))
        assert sorted((d for _, d in G2.degree()), reverse=True) == sorted(dseq, reverse=True), "own start lost its degree sequence"
    else:
        G2, reps = connect_graph(G.copy(), random.Random(sd * 31 + 5))
    assert nx.is_connected(G2)
    reps_log[sd] = reps
    env._cache[sd] = G2.copy(); env.G = G2; env.res = env.residuals(G2)
    return env.G, env.res, reps


class NoisedEnv(base.SwapEnv):

    def __init__(self, spec, edges, n, k):
        super().__init__(spec, forward_sweep=False)
        self.edges0 = [(int(u), int(v)) for u, v in edges]; self.n0 = int(n); self.k = int(k); self.info = {}

    def _start(self, seed):
        if seed in self._cache:
            return self._cache[seed].copy()
        G = nx.Graph(); G.add_nodes_from(range(self.n0)); G.add_edges_from(self.edges0)
        E0 = P.edge_set(G)
        done = P.noise_swaps(G, self.k, random.Random(7919 * seed + 17))
        self.info[seed] = dict(noise_k=self.k, noise_done=len(done),
                               start_edge_dist=round(1.0 - len(P.edge_set(G) & E0) / max(len(E0), 1), 4))
        self._cache[seed] = G.copy()
        return G


class CachedSwapEnv(base.SwapEnv):

    def __init__(self, spec, key, cache_dir):
        super().__init__(spec, forward_sweep=True)
        self.key = key; self.cache_dir = Path(cache_dir)

    @staticmethod
    def _load(f):
        d = json.loads(f.read_text()); G = nx.Graph()
        G.add_nodes_from(d["nodes"]); G.add_edges_from(tuple(e) for e in d["edges"])
        return G

    def _start(self, seed):
        if seed in self._cache:
            return self._cache[seed].copy()
        f = self.cache_dir / f"{self.key}_s{seed}.json"
        if not f.exists():
            G = super()._start(seed)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            import socket, uuid
            tmp = self.cache_dir / f".{f.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                tmp.write_text(json.dumps({"nodes": [int(v) for v in G.nodes()], "edges": [[int(u), int(v)] for u, v in G.edges()]}))
                chk = json.loads(tmp.read_text())
                if not chk["edges"] or len(chk["edges"]) != G.number_of_edges():
                    raise RuntimeError(f"warm-start cache write check failed for {f.name}")
                try:
                    os.link(tmp, f)
                except FileExistsError:
                    pass
            finally:
                tmp.unlink(missing_ok=True)
        G = self._load(f); self._cache[seed] = G.copy()
        return G


def make_ws_env(t, cfg):
    if getattr(cfg, "warm_cache", None):
        ident = {"m": {k: round(float(t[k]), 6) for k in ("clustering", "avg_path_length", "assortativity")},
                 "deg": [int(x) for x in t.get("degree_seq", [])],
                 "source": cfg.source, "warm": cfg.warm}
        key = hashlib.md5(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
        wc = Path(cfg.warm_cache)
        return CachedSwapEnv(spec_of(t), key, wc if wc.is_absolute() else _REPO / wc)
    return base.SwapEnv(spec_of(t), forward_sweep=True)


def make_eval_env(t, cfg):
    if cfg.warm == "noised":
        return NoisedEnv(spec_of(t), t["_graph_edges"], t["_n"], cfg.noise_k)
    if cfg.warm == "fam":
        return REAL.FamilyEnv(spec_of(t), t)
    return make_ws_env(t, cfg)


def parse_warm(v):
    if v in ("ws", "own", "fam"):
        return v, None
    m = re.fullmatch(r"noised:(\d+)", v)
    if m:
        return "noised", int(m.group(1))
    raise ValueError(f"--warm must be ws | own | fam | noised:K, got {v!r}")


def _spectral_gap_eigh(G):
    n = G.number_of_nodes()
    if n <= 2 or G.number_of_edges() == 0:
        return 0.0
    try:
        L = nx.normalized_laplacian_matrix(G).toarray().astype(np.float64)
        return float(max(0.0, np.linalg.eigvalsh(L)[1]))
    except Exception:
        return 0.0


def _spectral_gap_exact(G):
    n = G.number_of_nodes()
    if n <= 2 or G.number_of_edges() == 0:
        return 0.0
    try:
        if n <= 1500:
            return _spectral_gap_eigh(G)
        from scipy.sparse.linalg import eigsh as _eigsh
        L = nx.normalized_laplacian_matrix(G).astype(np.float64)
        vals = np.sort(np.real(_eigsh(L, k=3, sigma=-1e-6, which="LM", return_eigenvectors=False, tol=1e-8)))
        if abs(vals[0]) > 1e-6:
            return _spectral_gap_eigh(G)
        return float(max(0.0, vals[1]))
    except Exception:
        return 0.0


def _modularity_best(G):
    if G.number_of_edges() == 0:
        return 0.0
    try:
        from networkx.algorithms.community import greedy_modularity_communities, louvain_communities
        from networkx.algorithms.community.quality import modularity as nx_modularity
        H = nx.Graph(); H.add_nodes_from(sorted(G.nodes()))
        H.add_edges_from(sorted((u, v) if u < v else (v, u) for u, v in G.edges()))
        best = nx_modularity(H, list(greedy_modularity_communities(H)))
        for s in range(5):
            best = max(best, nx_modularity(H, louvain_communities(H, seed=s)))
        return float(best)
    except Exception:
        return 0.0


def set_oracle_q(mode):
    import warmstart.metrics as CM
    if mode == "cnm":
        CM.METRIC_FUNCTIONS["modularity"] = CM.modularity
    elif mode == "best":
        CM.METRIC_FUNCTIONS["modularity"] = _modularity_best
    else:
        raise ValueError(f"oracle_q must be cnm | best, got {mode!r}")


def set_nodes(n):
    n = int(n)
    if n == PT.N and n == base.N_NODES:
        return
    PT.N = n; PT.MIN_N = int(round(0.75 * n)); base.N_NODES = n
    log.info(f"[scale] N = {n} (targets.N, graph_env.N_NODES, MIN_N = {PT.MIN_N})")


METRIC_SETS = {"full5": None,
               "global3": ("avg_path_length", "modularity", "spectral_gap"),
               "local2": ("clustering", "assortativity")}


def set_metrics(name):
    if name not in METRIC_SETS:
        raise ValueError(f"--metrics must be one of {sorted(METRIC_SETS)}, got {name!r}")
    keep = METRIC_SETS[name]
    act = np.ones(len(base.KEYS), np.float32) if keep is None else np.array(
        [1.0 if k in keep else 0.0 for k in base.KEYS], np.float32)
    if keep is not None and act.sum() != len(keep):
        raise ValueError(f"metric set {name} does not match KEYS {base.KEYS}")
    base.ACTIVE = act
    if keep is not None:
        log.info(f"[metrics] {name} = {list(keep)} (scored {int(act.sum())}/{len(base.KEYS)}; others masked to 0)")


def _apl_igraph(G):
    if G.number_of_nodes() <= 1:
        return 0.0
    import igraph as ig
    lcc = max(nx.connected_components(G), key=len)
    if len(lcc) <= 1:
        return 0.0
    idx = {v: i for i, v in enumerate(sorted(lcc))}
    g = ig.Graph(n=len(idx), edges=[(idx[u], idx[v]) for u, v in G.subgraph(lcc).edges()])
    return float(g.average_path_length(directed=False, unconn=False))


def set_oracle_apl(mode):
    import warmstart.metrics as CM
    if mode == "nx":
        CM.METRIC_FUNCTIONS["avg_path_length"] = CM.avg_path_length
    elif mode == "igraph":
        CM.METRIC_FUNCTIONS["avg_path_length"] = _apl_igraph
    else:
        raise ValueError(f"oracle_apl must be nx | igraph, got {mode!r}")


def set_oracle_lam(mode):
    import warmstart.metrics as CM
    if mode == "eigsh":
        CM.METRIC_FUNCTIONS["spectral_gap"] = CM.spectral_gap
    elif mode == "eigh":
        CM.METRIC_FUNCTIONS["spectral_gap"] = _spectral_gap_eigh
    elif mode == "exact":
        CM.METRIC_FUNCTIONS["spectral_gap"] = _spectral_gap_exact
    else:
        raise ValueError(f"oracle_lam must be eigsh | eigh | exact, got {mode!r}")


_PSWAP_CACHE = {}
UCT_STATS = {"depth": __import__("collections").Counter()}


def eval_op(env, G, op):
    P.apply_op(G, op); r = env.residuals(G); P.revert_op(G, op)
    return r


def uct_choose(env, G, ops, res, groups, rng, cfg, weights=None, fam=None):
    if not ops:
        return None, 0
    W0 = max(worst(res), 1e-9)
    alpha = float(getattr(cfg, "uct_widen", 0.0) or 0.0); wc = float(getattr(cfg, "uct_widen_c", 1.0) or 1.0)

    def _may_expand(node):
        if not node["untried"]:
            return False
        if alpha <= 0:
            return True
        return len(node["kids"]) < max(1, int(math.ceil(wc * max(node["n"], 1) ** alpha)))

    def _node(cand, r):
        return {"ops": list(cand), "untried": list(range(len(cand))), "kids": {}, "n": 0, "w": 0.0, "res": r}

    root = _node(ops, res); used = 0
    while used < cfg.uct_expand:
        cur, ipath, nodes = root, [], [root]
        while not _may_expand(cur) and len(ipath) < cfg.uct_depth:
            live = [k for k in cur["kids"] if not cur["kids"][k].get("dead")]
            if not live:
                break
            ln = float(np.log(max(cur["n"], 1)))
            i = max(live, key=lambda k: cur["kids"][k]["w"] / max(cur["kids"][k]["n"], 1)
                    + cfg.uct_c * float(np.sqrt(ln / max(cur["kids"][k]["n"], 1))))
            ipath.append(i); cur = cur["kids"][i]; nodes.append(cur)
        if not cur["untried"] or len(ipath) >= cfg.uct_depth:
            cur["dead"] = True
            for nd_ in reversed(nodes[:-1]):
                if not nd_["untried"] and all(k_.get("dead") for k_ in nd_["kids"].values()):
                    nd_["dead"] = True
                else:
                    break
            if root.get("dead"):
                UCT_STATS["exhausted"] = UCT_STATS.get("exhausted", 0) + 1
                break
            continue
        UCT_STATS["depth"][len(ipath) + 1] += 1
        j = cur["untried"].pop(rng.randrange(len(cur["untried"]))); op = cur["ops"][j]
        nd, applied = root, []
        for i in ipath:
            P.apply_op(G, nd["ops"][i]); applied.append(nd["ops"][i]); nd = nd["kids"][i]
        r1 = eval_op(env, G, op); used += 1
        kid_ops = []
        if len(ipath) + 1 < cfg.uct_depth:
            P.apply_op(G, op)
            kid_ops = candidates(G, groups, r1, rng, cfg, acc={}, env=env, weights=weights, fam=fam)
            P.revert_op(G, op)
        for o in reversed(applied):
            P.revert_op(G, o)
        cur["kids"][j] = _node(kid_ops, r1)
        rw = float(np.clip((W0 - worst(r1)) / W0, -1.0, 1.0))
        nd = root; nd["n"] += 1; nd["w"] += rw
        for i in ipath:
            nd = nd["kids"][i]; nd["n"] += 1; nd["w"] += rw
        cur["kids"][j]["n"] += 1; cur["kids"][j]["w"] += rw
    if not root["kids"]:
        return rng.choice(ops), used
    best = max(root["kids"], key=lambda k: root["kids"][k]["w"] / max(root["kids"][k]["n"], 1))
    return root["ops"][best], used


class ValueAF(nn.Module):
    def __init__(self, in_dim, cfg, extra=P.AF_DIM):
        super().__init__(); self.cfg = cfg; self.enc = d0.Encoder(in_dim, cfg); e = cfg.emb; self.extra = extra
        self.head = nn.Sequential(nn.Linear(e + 5 + extra, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, An, Araw, X, res, af=None):
        h = self.enc(An, Araw, X); g = h.mean(0)
        parts = [g, res] + ([af] if (af is not None and self.extra > 0) else [])
        return self.head(torch.cat(parts, 0)).squeeze()


class ResOnlyNet(nn.Module):
    def __init__(self, extra=0):
        super().__init__(); self.extra = extra
        self.head = nn.Sequential(nn.Linear(5 + extra, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, An, Araw, X, res, af=None):
        parts = [res] + ([af] if (af is not None and self.extra > 0) else [])
        return self.head(torch.cat(parts, 0)).squeeze()


def label_of(kind, cfg):
    return LABEL_OF.get(kind) or cfg.value_label


def extra_dim(kind, cfg):
    af = (P.AF_DIM if kind == "value_af" else P.AF_DIM + SK.SKETCH_DIM if kind == "value_sketch"
          else 1 if kind == "value_sketch_min" else 0)
    return af + (FAM_DIM if cfg.value_fam else 0)


def make_net(kind, cfg):
    if kind == "value_rel_res":
        return ResOnlyNet(extra=FAM_DIM if cfg.value_fam else 0).to(DEV)
    return ValueAF(4, cfg.d0, extra=extra_dim(kind, cfg)).to(DEV)


def extra_vec(kind, cfg, af, fam):
    if kind == "value_af":
        parts = [np.asarray(af, np.float32)[:P.AF_DIM]]
    elif kind == "value_sketch":
        a = np.asarray(af, np.float32); assert a.shape[0] == P.AF_DIM + SK.SKETCH_DIM, "value_sketch needs sketch features"
        parts = [a]
    elif kind == "value_sketch_min":
        a = np.asarray(af, np.float32); assert a.shape[0] == P.AF_DIM + SK.SKETCH_DIM, "value_sketch_min needs sketch features"
        parts = [a[-1:]]
    else:
        parts = []
    parts += [fam_onehot(fam)] if cfg.value_fam else []
    return np.concatenate(parts).astype(np.float32) if parts else None


def sketch_af(G, op, sk, res, stats, lam_scale=1.0):
    af = P.op_features(G, op, stats)[0]
    return np.concatenate([af, SK.sketch_features(SK.cand_deltas(G, op, sk, res, lam_scale=lam_scale))]).astype(np.float32)


LAM_ARM_ALIAS = {"S_pswap": "S_swap"}


def resolve_uct_widen(v, c=1.0):
    v = str(v)
    if v.lower().startswith("auto:"):
        f = io_utils.RES / f"run_{v.split(':', 1)[1]}_sel.json"
        if not f.exists():
            raise FileNotFoundError(f"--uct-widen {v}: selection file {f} missing (selected on validation seeds; see README)")
        s = json.loads(f.read_text()); a_, c_ = float(s["uct_widen"]), float(s["uct_widen_c"])
        if not (0 < a_ <= 1 and 0 < c_ <= 8):
            raise ValueError(f"--uct-widen {v}: implausible selection {s}")
        return a_, c_
    return float(v), float(c)


def resolve_lam_scale(v, arm=None, nodes=200):
    if v is None:
        return 1.0
    v = str(v); tag = None
    if v.lower().startswith("auto:"):
        tag = v.split(":", 1)[1]; v = "auto"
    if v.lower() != "auto":
        return float(v)
    nodes = int(nodes or 200)
    if tag:
        p = io_utils.RES / f"run_{tag}_sketch_acc_metrics.csv"
        if not p.exists():
            raise RuntimeError(f"[sketch] lam_scale auto:{tag} — {p.name} missing (run the calibration + summary first)")
    else:
        p = io_utils.RES / ("run_s1_sketch_acc_metrics.csv" if nodes == 200 else f"run_p2n{nodes}_sketch_acc_metrics.csv")
    try:
        arm = LAM_ARM_ALIAS.get(arm, arm)
        m = pd.read_csv(p).set_index("metric"); key = f"spectral_gap@{arm}" if arm and f"spectral_gap@{arm}" in m.index else "spectral_gap"
        sl = float(m.loc[key, "slope"])
        if np.isfinite(sl) and 0.05 <= sl <= 2.0:
            log.info(f"[sketch] lam_scale auto = {sl:.3f} ({key}, {p.name})"); return sl
        if (nodes != 200 or tag) and key != "spectral_gap":
            sp = float(m.loc["spectral_gap", "slope"])
            if np.isfinite(sp) and 0.05 <= sp <= 2.0:
                log.warning(f"[sketch] lam_scale auto: implausible per-arm slope {sl} ({key}) → pooled spectral_gap {sp:.3f} ({p.name})"); return sp
        log.warning(f"[sketch] lam_scale auto: implausible slope {sl} ({key}) → 1.0")
    except Exception as e:
        if nodes != 200 or tag:
            raise RuntimeError(f"[sketch] lam_scale auto unavailable ({p.name}: {e}) — run the calibration first") from e
        log.warning(f"[sketch] lam_scale auto unavailable ({e}) → 1.0")
    if nodes != 200 or tag:
        raise RuntimeError(f"[sketch] lam_scale auto implausible ({p.name}) — run the calibration first")
    return 1.0


def _state_tensors(G, n, cfg):
    X = torch.tensor(d0.node_features(G, n, cfg), device=DEV)
    An = d0.norm_adj_dev(G, n); Araw = d0.dense_raw_adj(G, n) if cfg.encoder in ("gin", "gat") else An
    return An, Araw, X


def value_of(net, G, n, res, cfg, af=None):
    An, Araw, X = _state_tensors(G, n, cfg)
    with torch.no_grad():
        if af is None:
            return float(net.forward(An, Araw, X, torch.tensor(res, device=DEV)))
        return float(net.forward(An, Araw, X, torch.tensor(res, device=DEV), torch.tensor(af, device=DEV)))


def cand_score(net, G, n, r1, cfg, kind, af, fam):
    if kind == "value_rel_const":
        return worst(r1)
    v = value_of(net, G, n, r1, cfg.d0, extra_vec(kind, cfg, af, fam))
    return v * worst(r1) if label_of(kind, cfg) == "relative" else v


def beam_reachable_value(env, G0, band, groups, cfg, rng, fam=None):
    res0 = env.residuals(G0); best = float((np.abs(res0) / band[0]).max())
    if best <= 1.0:
        return best
    beam = [(G0.copy(), res0, best)]
    for _ in range(cfg.depth):
        children = []
        for (G, res, _) in beam:
            ops = candidates(G, groups, res, rng, cfg, env=env, fam=fam)
            if not ops:
                continue
            pool = ops if len(ops) <= cfg.k_expand else rng.sample(ops, cfg.k_expand)
            for op in pool:
                P.apply_op(G, op); r = env.residuals(G); sc = float((np.abs(r) / band[0]).max()); Gc = G.copy()
                P.revert_op(G, op); children.append((Gc, r, sc))
        if not children:
            break
        children.sort(key=lambda x: x[2]); best = min(best, children[0][2])
        if children[0][2] <= 1.0:
            return children[0][2]
        beam = children[:cfg.b]
    return best


def td_nstep_targets(rews, terms, boot, gamma, n, truncated=False):
    L = len(rews); out = []
    for t in range(L):
        if terms[t] or (truncated and t == L - 1):
            out.append(None); continue
        g = 0.0; disc = 1.0
        for k in range(1, n + 1):
            if t + k >= L:
                if truncated and not terms[L - 1]:
                    g += disc * boot[L - 1]
                break
            g += disc * rews[t + k]; disc *= gamma
            if terms[t + k]:
                break
            if k == n:
                g += disc * boot[t + k]
        out.append(g)
    return out


def td_q(net, G, n, res, r1, cfg, u):
    ws = float((np.abs(np.asarray(res)) / u).max()); w1s = float((np.abs(np.asarray(r1)) / u).max()); term = w1s <= 1.0
    rew = max(ws, 1.0) - max(w1s, 1.0) + (cfg.td_bonus if term else 0.0)
    v = 0.0 if (term or net is None) else value_of(net, G, n, r1, cfg.d0)
    return rew + cfg.td_gamma * v, rew, term


def train_value_td(kind, groups, ttargets, seed, cfg):
    u = get_band(TRAIN_BAND)[0]; nstep = cfg.td_n if kind == "value_td" else 10 ** 6
    torch.manual_seed(seed); net = None; oracle = 0; rng_t = np.random.default_rng(seed)
    envs = [make_ws_env(t, cfg) for t, _ in ttargets]
    for it in range(cfg.td_iters):
        t0 = time.time(); X = []; n_ep = 0; n_succ = 0
        for ti, (t, fam) in enumerate(ttargets):
            env = envs[ti]; sd = 1000 + seed * 100 + ti
            for ro in range(cfg.td_rollouts):
                rng = random.Random(sd * 19 + ro * 7 + it * 1009)
                G, res, _ = reset_connected(env, sd); G = G.copy(); n = G.number_of_nodes(); ep = []
                for _ in range(cfg.T):
                    ops = candidates(G, groups, res, rng, cfg, env=env, fam=fam)
                    if not ops:
                        break
                    pool = ops if len(ops) <= cfg.value_k else rng.sample(ops, cfg.value_k)
                    sc = []
                    for c in pool:
                        P.apply_op(G, c); r1 = env.residuals(G); oracle += 1
                        sc.append((*td_q(net, G, n, res, r1, cfg, u), r1)); P.revert_op(G, c)
                    i = rng.randrange(len(pool)) if rng.random() < cfg.td_eps else int(np.argmax([x[0] for x in sc]))
                    _, rew, term, r1 = sc[i]
                    P.apply_op(G, pool[i]); res = env.residuals(G)
                    ep.append((G.copy(), r1, rew, term))
                    if term:
                        break
                n_ep += 1; n_succ += int(bool(ep) and ep[-1][3])
                if ep:
                    boot = [0.0 if (e[3] or net is None) else value_of(net, e[0], e[0].number_of_nodes(), e[1], cfg.d0) for e in ep]
                    tg = td_nstep_targets([e[2] for e in ep], [e[3] for e in ep], boot, cfg.td_gamma, nstep, truncated=not ep[-1][3])
                    X += [(e[0], e[1], y) for e, y in zip(ep, tg) if y is not None]
        if net is None:
            net = make_net(kind, cfg)
        if X:
            opt = torch.optim.Adam(net.parameters(), lr=cfg.lr); ls = []
            for step in range(cfg.td_steps):
                G_, r1_, y = X[int(rng_t.integers(len(X)))]
                An, Araw, Xf = _state_tensors(G_, G_.number_of_nodes(), cfg.d0)
                loss = (net.forward(An, Araw, Xf, torch.tensor(r1_, device=DEV), None) - torch.tensor(float(y), device=DEV)) ** 2
                opt.zero_grad(); loss.backward(); opt.step(); ls.append(float(loss))
            log.info(f"    [{kind}] iter{it}: {n_ep} episodes ({n_succ} reached u), {len(X)} targets (mean {np.mean([x[2] for x in X]):.3f}), "
                     f"loss last-200 {np.mean(ls[-200:]):.4f} ({time.time()-t0:.0f}s)")
    return net, oracle


def collect_value_data(groups, arm_id, band, targets, seed, cfg):
    data = []
    for ti, (t, fam) in enumerate(targets):
        spec = spec_of(t); env = make_ws_env(t, cfg)
        for ro in range(cfg.rollouts):
            sd = 1000 + seed * 100 + ti; rng = random.Random(sd * 13 + ro + 97 * arm_id)
            G, res, _ = reset_connected(env, sd); G = G.copy(); mode = "rand" if ro % 2 == 0 else "gw"
            af = np.zeros(P.AF_DIM + (SK.SKETCH_DIM if cfg.need_sketch else 0), np.float32)
            rng_sk = random.Random(sd * 7 + ro + 1)
            for step in range(cfg.rollout_len):
                tv = beam_reachable_value(env, G, band, groups, cfg, rng, fam=fam)
                data.append((G.copy(), res.copy(), af.copy(), tv, fam, env))
                if float((np.abs(res) / band[0]).max()) <= 1.0:
                    break
                ops = candidates(G, groups, res, rng, cfg, env=env, fam=fam)
                if not ops:
                    break
                if mode == "rand":
                    op = rng.choice(ops)
                else:
                    pool = ops if len(ops) <= cfg.value_k else rng.sample(ops, cfg.value_k)
                    op, bsc = pool[0], 1e9
                    for cand in pool:
                        sc = float((np.abs(eval_op(env, G, cand)) / band[0]).max())
                        if sc < bsc:
                            bsc, op = sc, cand
                S = P.degree_stats(G)
                if cfg.need_sketch:
                    sk = SK.state_sketch(G, G.number_of_nodes(), rng_sk, k_src=cfg.sketch_k); af = sketch_af(G, op, sk, res, S, cfg.sketch_lam_scale)
                else:
                    af, _ = P.op_features(G, op, S)
                P.apply_op(G, op); res = env.residuals(G)
    return data


def train_value_net(kind, data, cfg, seed):
    torch.manual_seed(seed); d0c = cfg.d0
    net = make_net(kind, cfg)
    if not data:
        return net
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr); rng = np.random.default_rng(seed); t0 = time.time()
    lab = label_of(kind, cfg)
    if lab == "clip3":
        tgt = np.array([min(d[3], 3.0) for d in data], np.float32)
    elif lab == "noclip":
        tgt = np.array([d[3] for d in data], np.float32)
    elif lab == "relative":
        u = float(TRAIN_BAND[1:])
        tgt = np.array([min(d[3] / max(worst(d[1]) / u, 1e-3), 1.0) for d in data], np.float32)
    else:
        raise ValueError(lab)
    n_steps = cfg.value_epochs
    for ep in range(n_steps):
        idx = rng.integers(len(data), size=cfg.value_batch)
        loss = 0.0
        for i in idx:
            G, res, af, _, fam = data[int(i)][:5]; n = G.number_of_nodes()
            An, Araw, X = _state_tensors(G, n, d0c)
            ev = extra_vec(kind, cfg, af, fam)
            pred = net.forward(An, Araw, X, torch.tensor(res, device=DEV), None if ev is None else torch.tensor(ev, device=DEV))
            loss = loss + (pred - torch.tensor(float(tgt[int(i)]), device=DEV)) ** 2
        loss = loss / len(idx)
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 400 == 0:
            log.info(f"    [{kind}] step{ep} loss={float(loss):.3f} ({time.time()-t0:.0f}s)")
    return net


class LocalFit:
    def __init__(self, coef, mean, std):
        self.coef, self.mean, self.std = coef, mean, std

    def predict(self, F):
        Z = (np.asarray(F, np.float32) - self.mean) / self.std
        return Z @ self.coef[:-1] + self.coef[-1]


def fit_ridge(F, y, lam=1e-2):
    F = np.asarray(F, np.float64); y = np.asarray(y, np.float64)
    mean = F.mean(0); std = F.std(0) + 1e-6; Z = (F - mean) / std
    Zb = np.hstack([Z, np.ones((len(Z), 1))])
    A = Zb.T @ Zb + lam * np.eye(Zb.shape[1]); A[-1, -1] -= lam
    coef = np.linalg.solve(A, Zb.T @ y)
    return LocalFit(coef.astype(np.float32), mean.astype(np.float32), std.astype(np.float32))


def count_env(env):
    if getattr(env, "_counted", False):
        return env
    f = env.residuals
    def g(G):
        env.n_eval = getattr(env, "n_eval", 0) + 1; t0_ = time.perf_counter()
        try:
            return f(G)
        finally:
            env.t_eval = getattr(env, "t_eval", 0.0) + (time.perf_counter() - t0_)
    env._res_raw = f
    env.residuals = g; env._counted = True; env.n_eval = getattr(env, "n_eval", 0); env.t_eval = getattr(env, "t_eval", 0.0)
    return env


def rollout(policy, model, env, groups, sd, cfg, fam=None, weights=None):
    t_w0 = time.perf_counter(); n_ev0 = getattr(env, "n_eval", 0); t_ev0 = getattr(env, "t_eval", 0.0)
    G, res, reps = reset_connected(env, sd); G = G.copy(); n = G.number_of_nodes(); rng = random.Random(sd)
    start_hash = graph_hash(G)
    traj = [res.copy()]; oracle = 0; chosen = []; d0c = cfg.d0; kbar0 = P.kbar(G); acc = {}; mix_ms = []
    orc_cum = [0]
    tot_cum = [getattr(env, "n_eval", 0) - n_ev0]; wall_cum = [time.perf_counter() - t_w0]
    need_sk = policy in SKETCH_POLICIES or policy in ("value_sketch", "value_sketch_min") or policy in ("preds", "preds_k8") or EXT.needs_sketch(policy)
    ext_st = EXT.init_state(policy, sd, cfg, model) if EXT.is_ext(policy) else None
    stop = get_band(cfg.stop_band) if getattr(cfg, "stop_band", None) else None
    rng_sk = random.Random(sd * 7 + 1); sk = None; sk_step = -10 ** 9; sketch_calls = 0; sketch_cand = 0; sketch_time = 0.0
    bind0 = base.KEYS[int(np.argmax(np.abs(res)))]
    bind_cnt = {k: 0 for k in base.KEYS}
    g_hit = None; band_hit = get_band(SAVE_BAND)
    if cfg.save_graphs and TB.sat_band(res, band_hit):
        g_hit = list(G.edges())
    B_max = getattr(cfg, "max_oracle", None); budget_stop = 0
    for _ in range(cfg.T):
        if stop is not None and TB.sat_band(res, stop):
            break
        if B_max is not None and getattr(env, "n_eval", 0) - n_ev0 >= B_max:
            budget_stop = 1; break
        bind_cnt[base.KEYS[int(np.argmax(np.abs(res)))]] += 1
        if policy in MIXB_POLICIES:
            m_c = mixb_choose(model, res, cfg); mix_ms.append(m_c); weights = mixb_weights(groups, m_c, cfg.K)
        elif fixm_of(policy) is not None:
            m_c = fixm_of(policy); mix_ms.append(m_c); weights = mixb_weights(groups, m_c, cfg.K)
        ops, fied = candidates(G, groups, res, rng, cfg, with_fied=True, acc=acc, env=env, weights=weights, fam=fam)
        if not ops:
            continue
        if need_sk:
            t_sk = time.perf_counter(); step_i = len(chosen)
            comm = sk["comm"] if (sk is not None and step_i - sk_step < cfg.sketch_q_refresh) else None
            if comm is None:
                sk_step = step_i
            sk = SK.state_sketch(G, n, rng_sk, k_src=cfg.sketch_k, comm=comm); sketch_calls += 1
            sketch_time += time.perf_counter() - t_sk
        r_sel = None
        if selector_of(policy) == "random":
            op = rng.choice(ops)
        elif policy in SKETCH_POLICIES:
            t_sk = time.perf_counter()
            what = [(SK.cand_deltas(G, cand, sk, res, lam_scale=cfg.sketch_lam_scale)["W_hat"], i) for i, cand in enumerate(ops)]; sketch_cand += len(ops)
            sketch_time += time.perf_counter() - t_sk
            what.sort()
            if policy == "sketch":
                op = ops[what[0][1]]
            else:
                op, bsc = None, 1e18
                for _, i in what[:cfg.value_k]:
                    r1 = eval_op(env, G, ops[i]); oracle += 1; sc = worst(r1)
                    if sc < bsc:
                        bsc, op, r_sel = sc, ops[i], r1
        elif policy in PRED_POLICIES:
            S_ = P.degree_stats(G); w0 = worst(res)
            Fl = [P.ext_features(G, cand, S_, fied, w0)[0] for cand in ops]; rh = What_ = None
            if policy.startswith("preds"):
                t_sk = time.perf_counter()
                cds = [SK.cand_deltas(G, cand, sk, res, lam_scale=cfg.sketch_lam_scale) for cand in ops]; sketch_cand += len(ops)
                rh = np.array([c_["res_hat"] for c_ in cds], np.float32); What_ = np.array([c_["W_hat"] for c_ in cds], np.float32)
                sketch_time += time.perf_counter() - t_sk
            if policy.startswith("gpred"):
                order = np.argsort(model.predict_cands(G, ops, Fl, res), kind="stable")
            else:
                order = np.argsort(model.predict(PR.build_X(Fl, res, w0, rh, What_)), kind="stable")
            if policy in ("pred", "preds", "gpred"):
                op = ops[int(order[0])]
            else:
                op, bsc = None, 1e18
                for i in order[:cfg.value_k]:
                    r1 = eval_op(env, G, ops[int(i)]); oracle += 1; sc = worst(r1)
                    if sc < bsc:
                        bsc, op, r_sel = sc, ops[int(i)], r1
        elif ext_st is not None:
            op, r_sel, _n = EXT.choose(policy, ext_st, G, ops, fied, res, env, rng, cfg, sk, worst); oracle += _n
        elif policy in UCT_POLICIES:
            if not chosen:
                UCT_STATS["depth"].clear()
            op, _u = uct_choose(env, G, ops, res, groups, rng, cfg, weights=weights, fam=fam); oracle += _u
        elif selector_of(policy) == "greedy":
            pool = ops if len(ops) <= cfg.greedy_k else rng.sample(ops, cfg.greedy_k)
            op, bsc = None, -1e18
            for cand in pool:
                r1 = eval_op(env, G, cand); oracle += 1
                sc = base.phi(r1) if cfg.greedy_obj == "phi" else -worst(r1)
                if sc > bsc:
                    bsc, op, r_sel = sc, cand, r1
        elif policy in VALUE_POLICIES:
            pool = ops if len(ops) <= cfg.value_k else rng.sample(ops, cfg.value_k)
            S = P.degree_stats(G) if policy in ("value_af", "value_sketch", "value_sketch_min") else None
            op, bv = None, 1e18
            for cand in pool:
                if policy == "value_af":
                    af = P.op_features(G, cand, S)[0]
                elif policy in ("value_sketch", "value_sketch_min"):
                    t_sk = time.perf_counter(); af = sketch_af(G, cand, sk, res, S, cfg.sketch_lam_scale); sketch_cand += 1; sketch_time += time.perf_counter() - t_sk
                else:
                    af = None
                P.apply_op(G, cand); r1 = env.residuals(G); oracle += 1
                v = (-td_q(model, G, n, res, r1, cfg, get_band(TRAIN_BAND)[0])[0]) if policy in TD_KINDS else cand_score(model, G, n, r1, cfg, policy, af, fam)
                P.revert_op(G, cand)
                if v < bv:
                    bv, op, r_sel = v, cand, r1
            if op is None:
                op = pool[0]; r_sel = None
        elif policy == "localfit":
            S = P.degree_stats(G); w0 = worst(res)
            F = [P.ext_features(G, cand, S, fied, w0)[0] for cand in ops]
            op = ops[int(np.argmin(model.predict(F)))]
        else:
            raise ValueError(policy)
        P.apply_op(G, op); res = r_sel if r_sel is not None else env.residuals(G); traj.append(res.copy()); chosen.append(op); orc_cum.append(oracle)
        tot_cum.append(getattr(env, "n_eval", 0) - n_ev0); wall_cum.append(time.perf_counter() - t_w0)
        if cfg.save_graphs and g_hit is None and TB.sat_band(res, band_hit):
            g_hit = list(G.edges())
    nreq = max(acc.get("requested", 1), 1); nsteps = max(cfg.T, 1)
    mixb_cols = (dict(mixb_m_mean=round(float(np.mean(mix_ms)), 3), mixb_frac_hi=round(float(np.mean([m_ >= 24 for m_ in mix_ms])), 3))
                 if mix_ms else {})
    return dict(**mixb_cols, traj=traj, oracle=oracle, oracle_cum=orc_cum, tot_cum=tot_cum, wall_cum=wall_cum,
                g_hit=g_hit, g_final=(list(G.edges()) if cfg.save_graphs else None),
                oracle_total=tot_cum[-1], t_wall=round(time.perf_counter() - t_w0, 3), oracle_time=round(getattr(env, "t_eval", 0.0) - t_ev0, 3),
                start_hash=start_hash, kbar_start=kbar0, kbar_final=P.kbar(G), best_worst=min(worst(r) for r in traj), bind0=bind0,
                ws_repairs=reps, screen_calls=acc.get("screen_calls", 0),
                sketch_calls=sketch_calls, sketch_cand=sketch_cand, sketch_time=round(sketch_time, 3),
                pool_overlap=round(acc.get("div_sum", 0.0) / max(acc.get("div_n", 1), 1), 4),
                steps=len(chosen), k_eff=round(acc.get("produced", 0) / max(acc.get("steps", 1), 1), 2),
                fallback_rate=round(acc.get("fallback", 0) / nreq, 4), mask_rate=round(acc.get("masked", 0) / nreq, 4),
                **{f"bind_frac_{k}": round(bind_cnt[k] / nsteps, 3) for k in base.KEYS},
                **({f"uct_d{d}": int(UCT_STATS["depth"].get(d, 0)) for d in range(1, max(int(cfg.uct_depth), 1) + 1)} if policy in UCT_POLICIES else {}),
                **(EXT.summary(ext_st) if ext_st is not None else {}),
                **({"ext_budget_stop": budget_stop} if B_max is not None else {}),
                **P.op_usage(chosen))


def sa_temperature(policy, i, B, cfg):
    if policy in ("sa", "sa_free"):
        return float(cfg.sa_tau0 * (cfg.sa_tau1 / cfg.sa_tau0) ** (i / max(B - 1, 1)))
    if policy == "mcmc":
        return float(cfg.sa_tau_fixed)
    if SA_UNCAPPED.get(policy) is not None:
        return float(SA_UNCAPPED[policy])
    return 0.0


def sa_rollout(policy, env, groups, sd, cfg, fam=None, weights=None):
    t_w0 = time.perf_counter(); n_ev0 = getattr(env, "n_eval", 0); t_ev0 = getattr(env, "t_eval", 0.0)
    G, res, reps = reset_connected(env, sd); G = G.copy(); n = G.number_of_nodes(); rng = random.Random(sd)
    start_hash = graph_hash(G); B = int(cfg.sa_budget) if cfg.sa_budget else int(cfg.greedy_k * cfg.T)
    traj = [res.copy()]; oracle = 0; chosen = []; orc_cum = [0]; acc = {}; kbar0 = P.kbar(G); pool = []
    tot_cum = [getattr(env, "n_eval", 0) - n_ev0]; wall_cum = [time.perf_counter() - t_w0]
    bind0 = base.KEYS[int(np.argmax(np.abs(res)))]; bind_cnt = {k: 0 for k in base.KEYS}
    W = worst(res); n_prop = 0; n_acc = 0; stalls = 0; stop_reason = "budget"
    g_hit = None; band_hit = get_band(SAVE_BAND)
    if cfg.save_graphs and TB.sat_band(res, band_hit):
        g_hit = list(G.edges())
    while n_prop < B and (policy in SA_UNCAPPED or n_acc < cfg.T):
        if not pool:
            ops = candidates(G, groups, res, rng, cfg, acc=acc, env=env, weights=weights, fam=fam)
            if not ops:
                stalls += 1
                if stalls >= 3:
                    stop_reason = "no_candidates"; break
                continue
            stalls = 0; pool = list(ops); rng.shuffle(pool)
        op = pool.pop()
        r1 = eval_op(env, G, op); oracle += 1; n_prop += 1; W1 = worst(r1); dW = W1 - W
        tau = sa_temperature(policy, n_prop - 1, B, cfg)
        if dW <= 0 or (tau > 0 and rng.random() < math.exp(-dW / tau)):
            bind_cnt[base.KEYS[int(np.argmax(np.abs(res)))]] += 1
            P.apply_op(G, op); res = r1; W = W1; n_acc += 1; pool = []
            traj.append(res.copy()); chosen.append(op); orc_cum.append(oracle)
            tot_cum.append(getattr(env, "n_eval", 0) - n_ev0); wall_cum.append(time.perf_counter() - t_w0)
            if cfg.save_graphs and g_hit is None and TB.sat_band(res, band_hit):
                g_hit = list(G.edges())
    if stop_reason == "budget" and n_prop < B:
        stop_reason = "edit_cap"
    nreq = max(acc.get("requested", 1), 1); nsteps = max(n_acc, 1)
    return dict(traj=traj, oracle=oracle, oracle_cum=orc_cum, tot_cum=tot_cum, wall_cum=wall_cum,
                g_hit=g_hit, g_final=(list(G.edges()) if cfg.save_graphs else None),
                oracle_total=getattr(env, "n_eval", 0) - n_ev0, sa_stop=stop_reason, t_wall=round(time.perf_counter() - t_w0, 3), oracle_time=round(getattr(env, "t_eval", 0.0) - t_ev0, 3), start_hash=start_hash, kbar_start=kbar0, kbar_final=P.kbar(G),
                best_worst=min(worst(r) for r in traj), bind0=bind0, ws_repairs=reps, screen_calls=acc.get("screen_calls", 0),
                sketch_calls=0, sketch_cand=0, sketch_time=0.0,
                pool_overlap=round(acc.get("div_sum", 0.0) / max(acc.get("div_n", 1), 1), 4),
                steps=n_acc, k_eff=round(acc.get("produced", 0) / max(acc.get("steps", 1), 1), 2),
                fallback_rate=round(acc.get("fallback", 0) / nreq, 4), mask_rate=round(acc.get("masked", 0) / nreq, 4),
                sa_proposals=n_prop, sa_accepted=n_acc, sa_accept_rate=round(n_acc / max(n_prop, 1), 4), sa_budget=B,
                **{f"bind_frac_{k}": round(bind_cnt[k] / nsteps, 3) for k in base.KEYS},
                **P.op_usage(chosen))


LAST_HIT_GRAPH = {"edges": None}
SAVE_BAND = "u0.4"


def beam_rollout(env, groups, sd, cfg, band, net=None, kind="value_rel", fam=None, acc=None):
    n_ev0 = getattr(env, "n_eval", 0)
    G0, res0, _ = reset_connected(env, sd); G0 = G0.copy(); rng = random.Random(sd); oracle = 0; n = G0.number_of_nodes()
    acc = {} if acc is None else acc
    bw = float(worst(res0)); acc.update(beam_best_worst=round(bw, 4), beam_depth_hit=np.nan, beam_depth=0)
    LAST_HIT_GRAPH["edges"] = None
    if TB.sat_band(res0, band):
        LAST_HIT_GRAPH["edges"] = list(G0.edges())
        acc.update(beam_depth_hit=0); return 1, 0
    beam = [(G0, res0, float((np.abs(res0) / band[0]).max()), 0.0)]
    B_max = getattr(cfg, "max_oracle", None)
    for depth in range(1, cfg.T + 1):
        if B_max is not None and getattr(env, "n_eval", 0) - n_ev0 >= B_max:
            break
        children = []
        for (G, res, _, _) in beam:
            ops = candidates(G, groups, res, rng, cfg, env=env, fam=fam)
            if not ops:
                continue
            pool = ops if len(ops) <= cfg.k_expand else rng.sample(ops, cfg.k_expand)
            for op in pool:
                P.apply_op(G, op); r = env.residuals(G); oracle += 1; sc = float((np.abs(r) / band[0]).max())
                rk = cand_score(net, G, n, r, cfg, kind, None, fam) if net is not None else sc
                Gc = G.copy(); P.revert_op(G, op); children.append((Gc, r, sc, rk)); bw = min(bw, float(worst(r)))
        if not children:
            break
        acc.update(beam_best_worst=round(bw, 4), beam_depth=depth)
        if min(c[2] for c in children) <= 1.0:
            LAST_HIT_GRAPH["edges"] = list(min(children, key=lambda c: c[2])[0].edges())
            acc.update(beam_depth_hit=depth); return 1, oracle
        children.sort(key=lambda x: x[3])
        beam = children[:cfg.b]
    LAST_HIT_GRAPH["edges"] = None
    return 0, oracle


def beam_sketch_rollout(env, groups, sd, cfg, band, fam=None, acc=None):
    n_ev0 = getattr(env, "n_eval", 0)
    G0, res0, _ = reset_connected(env, sd); G0 = G0.copy(); rng = random.Random(sd); oracle = 0; n = G0.number_of_nodes()
    rng_sk = random.Random(sd * 7 + 1); skc = 0; skn = 0; skt = 0.0
    acc = {} if acc is None else acc
    LAST_HIT_GRAPH["edges"] = None
    bw = float(worst(res0)); acc.update(beam_best_worst=round(bw, 4), beam_depth_hit=np.nan, beam_depth=0)
    if TB.sat_band(res0, band):
        LAST_HIT_GRAPH["edges"] = list(G0.edges())
        acc.update(sketch_calls=0, sketch_cand=0, sketch_time=0.0, beam_depth_hit=0); return 1, 0
    beam = [(G0, res0, None, 0)]
    B_max = getattr(cfg, "max_oracle", None)
    for depth in range(1, cfg.T + 1):
        if B_max is not None and getattr(env, "n_eval", 0) - n_ev0 >= B_max:
            break
        scored = []; part = {}
        for bi, (G, res, comm, age) in enumerate(beam):
            ops = candidates(G, groups, res, rng, cfg, env=env, fam=fam)
            if not ops:
                continue
            reuse = comm if (comm is not None and age < cfg.sketch_q_refresh) else None
            t_sk = time.perf_counter()
            sk = SK.state_sketch(G, n, rng_sk, k_src=cfg.sketch_k, comm=reuse); skc += 1
            part[bi] = (sk["comm"], age if reuse is not None else 0)
            for op in ops:
                scored.append((SK.cand_deltas(G, op, sk, res, lam_scale=cfg.sketch_lam_scale)["W_hat"], len(scored), bi, op)); skn += 1
            skt += time.perf_counter() - t_sk
        if not scored:
            break
        scored.sort(key=lambda x: (x[0], x[1]))
        children = []
        for _, _, bi, op in scored[:cfg.b]:
            G = beam[bi][0]
            P.apply_op(G, op); r = env.residuals(G); oracle += 1
            Gc = G.copy(); P.revert_op(G, op); children.append((Gc, r, part[bi][0], part[bi][1] + 1))
            bw = min(bw, float(worst(r)))
        acc.update(beam_best_worst=round(bw, 4), beam_depth=depth)
        if min(float((np.abs(c_[1]) / band[0]).max()) for c_ in children) <= 1.0:
            LAST_HIT_GRAPH["edges"] = list(min(children, key=lambda c_: float((np.abs(c_[1]) / band[0]).max()))[0].edges())
            acc.update(sketch_calls=skc, sketch_cand=skn, sketch_time=round(skt, 3), beam_depth_hit=depth); return 1, oracle
        beam = children
    LAST_HIT_GRAPH["edges"] = None
    acc.update(sketch_calls=skc, sketch_cand=skn, sketch_time=round(skt, 3))
    return 0, oracle


MIXB_M = (0, 4, 8, 24, 28, 32)
MIXB_POLICIES = ("greedy_mixb", "random_mixb")
MIXB_NCTX = 15


def fixm_of(policy):
    m = re.fullmatch(r"(greedy|random)_fixm(\d+)", policy)
    if not m:
        return None
    k = int(m.group(2))
    if not 0 <= k <= 32:
        raise ValueError(f"{policy}: typed count must be in 0..32")
    return k


def selector_of(policy):
    return policy.split("_")[0] if (policy in MIXB_POLICIES or fixm_of(policy) is not None) else policy


def mixb_context(res):
    a = np.abs(np.asarray(res, np.float64)); w = float(a.max())
    return int(np.argmax(a)) * 3 + (0 if w >= 2.0 else (1 if w >= 1.0 else 2))


def mixb_weights(groups, m, K):
    if len(groups) != 2 or groups[1] != "t" + groups[0]:
        raise ValueError(f"mixb needs an arm of the form g+tg, got {groups}")
    mk = int(round(m * K / 32.0))
    return {groups[0]: K - mk, groups[1]: mk}


def mixb_choose(table, res, cfg, rng=None, eps=0.0):
    if rng is not None and rng.random() < eps:
        return MIXB_M[rng.randrange(len(MIXB_M))]
    S, N = table["sum"], table["cnt"]; c = mixb_context(res); z = c % 3
    same = [c_ for c_ in range(S.shape[0]) if c_ % 3 == z]
    bn, bs, gn, gs = N[same].sum(0), S[same].sum(0), N.sum(0), S.sum(0)
    prior = np.where(bn > 0, bs / np.maximum(bn, 1), np.where(gn > 0, gs / np.maximum(gn, 1), np.nan))
    k = float(cfg.mixb_min_count)
    mean = np.where(np.isnan(prior), -np.inf, (S[c] + k * np.nan_to_num(prior)) / (N[c] + k))
    if not np.isfinite(mean).any():
        return MIXB_M[rng.randrange(len(MIXB_M))] if rng is not None else MIXB_M[-1]
    return MIXB_M[int(np.argmax(mean))]


def _mixb_state(res_like_ctx):
    b, z = divmod(res_like_ctx, 3); v = np.zeros(5); v[b] = (3.0, 1.5, 0.5)[z]
    return v


def save_mixb_table(tb, path, cfg):
    recs = []
    for c in range(tb["sum"].shape[0]):
        m_star = mixb_choose(tb, _mixb_state(c), cfg); visits = int(tb["cnt"][c].sum())
        for i, m_ in enumerate(MIXB_M):
            n_ = tb["cnt"][c, i]
            recs.append(dict(context=c, m=m_, sum=round(float(tb["sum"][c, i]), 6), cnt=int(n_),
                             mean=(round(float(tb["sum"][c, i] / n_), 6) if n_ else np.nan), m_star=m_star, ctx_visits=visits,
                             train_oracle=int(tb["train_oracle"])))
    tmp = Path(str(path) + f".tmp{os.getpid()}")
    pd.DataFrame(recs).to_csv(tmp, index=False); os.replace(tmp, path)


def load_mixb_table(path):
    d = pd.read_csv(path); S = np.zeros((MIXB_NCTX, len(MIXB_M))); N = np.zeros_like(S)
    for r in d.itertuples():
        S[r.context, MIXB_M.index(r.m)] = r.sum; N[r.context, MIXB_M.index(r.m)] = r.cnt
    return dict(sum=S, cnt=N, train_oracle=int(d.train_oracle.iloc[0]))


def train_mixb(groups, ttargets, seed, cfg):
    S = np.zeros((MIXB_NCTX, len(MIXB_M))); N = np.zeros_like(S); table = dict(sum=S, cnt=N); oracle = 0
    mixb_weights(groups, 0, cfg.K)
    for ti, (t, fam) in enumerate(ttargets):
        env = make_ws_env(t, cfg); sd = 5000 + seed * 100 + ti
        for ro in range(cfg.mixb_rollouts):
            rng = random.Random(sd * 17 + ro); rb = random.Random(sd * 29 + ro + 7)
            G, res, _ = reset_connected(env, sd); G = G.copy()
            for _ in range(cfg.T):
                m_ = mixb_choose(table, res, cfg, rng=rb, eps=cfg.mixb_eps); c = mixb_context(res)
                ops = candidates(G, groups, res, rng, cfg, env=env, weights=mixb_weights(groups, m_, cfg.K), fam=fam)
                if not ops:
                    continue
                pool = ops if len(ops) <= cfg.greedy_k else rng.sample(ops, cfg.greedy_k)
                op, bsc = None, 1e18
                for cand in pool:
                    sc = worst(eval_op(env, G, cand)); oracle += 1
                    if sc < bsc:
                        bsc, op = sc, cand
                S[c, MIXB_M.index(m_)] += worst(res) - bsc; N[c, MIXB_M.index(m_)] += 1
                P.apply_op(G, op); res = env.residuals(G)
    table["train_oracle"] = oracle
    return table


def out_dir(tag):
    d = io_utils.RES / f"{EXP}_{tag}"; d.mkdir(parents=True, exist_ok=True); return d


def arm_id(arm):
    return sum((i + 1) * ord(c) for i, c in enumerate(arm)) % 100_003


def targets_for(cfg, seed):
    tc = getattr(cfg, "target_cache", None)
    if not tc:
        return _targets_for_raw(cfg, seed)
    if cfg.warm != "ws":
        raise ValueError("--target-cache supports warm=ws only (own/noised target dicts carry graphs)")
    ident = dict(source=cfg.source, seed=int(seed), train=int(cfg.train_targets), eval=int(cfg.eval_targets),
                 families=list(cfg.families) if getattr(cfg, "families", None) else None,
                 oracle_lam=getattr(cfg, "oracle_lam", "eigsh"), oracle_q=getattr(cfg, "oracle_q", "cnm"))
    key = hashlib.md5(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    d = Path(tc); d = d if d.is_absolute() else _REPO / d; f = d / f"targets_{key}.json"
    if not f.exists():
        tr, ev = _targets_for_raw(cfg, seed)
        d.mkdir(parents=True, exist_ok=True)
        import socket, uuid
        tmp = d / f".{f.name}.{socket.gethostname()}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            tmp.write_text(json.dumps(dict(ident=ident, train=[[t, fam] for t, fam in tr], eval=[[t, fam] for t, fam in ev]), default=float))
            try:
                os.link(tmp, f)
            except FileExistsError:
                pass
        finally:
            tmp.unlink(missing_ok=True)
    js = json.loads(f.read_text())
    if js.get("ident") != ident:
        raise RuntimeError(f"target cache {f.name}: identity mismatch {js.get('ident')} != {ident}")
    return [(dict(t), fam) for t, fam in js["train"]], [(dict(t), fam) for t, fam in js["eval"]]


def _targets_for_raw(cfg, seed):
    if str(cfg.source).startswith("real:"):
        ev = REAL.parse_real_source(cfg.source)
        if cfg.warm == "own":
            ev = [({**t, "degree_seq": t["_degseq"]}, lab) for t, lab in ev]
        return ev, ev
    if cfg.warm in ("own", "noised") and cfg.source != "mix7":
        raise ValueError(f"warm start '{cfg.warm}' is implemented for source=mix7 only")
    if cfg.source == "ws":
        tr = [(t, "ws") for t in e37.sample_targets(cfg.train_targets, seed=1000 + seed)]
        ev = [(t, "ws") for t in e37.sample_targets(cfg.eval_targets, seed=7000 + seed)]
        return tr, ev
    if cfg.source == "mix7":
        fams = cfg.families if getattr(cfg, "families", None) else PT.FAMILIES
        own = (cfg.warm == "own")
        return (PT.sample_targets(cfg.train_targets, 1000 + seed, families=fams, with_degseq=own),
                PT.sample_targets(cfg.eval_targets, 7000 + seed, families=fams, with_degseq=own, with_graph=(cfg.warm == "noised")))
    raise ValueError(f"unknown target source {cfg.source}")


CKPT_KEYS = ("value_label", "value_fam", "value_epochs", "value_batch", "lr", "rollouts", "rollout_len", "depth", "b", "k_expand",
             "value_k", "K", "T", "kband", "mask_connected", "train_targets", "source", "families", "arm_weights", "oracle_lam", "oracle_q",
             "screen_mult", "pswap_model")
TD_CKPT_KEYS = ("td_gamma", "td_bonus", "td_n", "td_eps", "td_iters", "td_rollouts", "td_steps")


def _norm_cfg(v):
    if isinstance(v, (list, tuple)):
        return [_norm_cfg(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _norm_cfg(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, float):
        return round(v, 9)
    return v


def ckpt_mismatch(kind, meta, cfg, own_tag=True):
    keys = list(CKPT_KEYS)
    if kind in ("value_sketch", "value_sketch_min") or kind in HORIZON_OF:
        keys += ["sketch_k", "sketch_lam_scale"]
    if kind in TD_KINDS:
        keys += list(TD_CKPT_KEYS)
    if own_tag:
        keys += ["warm", "noise_k"]
    cur = cfg.as_dict(); meta = meta or {}
    bad = {k: (meta[k], cur[k]) for k in keys if k in meta and k in cur and _norm_cfg(meta[k]) != _norm_cfg(cur[k])}
    if "arm_weights" not in meta and cur.get("arm_weights") is not None:
        bad["arm_weights"] = ("(absent)", cur["arm_weights"])
    return bad


def get_or_train(kind, arm, groups, seed, tag, cfg, ttargets, band, cache):
    ck = io_utils.load_checkpoint(f"{EXP}_{tag}", kind, seed, arm); src = tag
    if ck is None and cfg.ckpt_tag:
        ck = io_utils.load_checkpoint(f"{EXP}_{cfg.ckpt_tag}", kind, seed, arm); src = cfg.ckpt_tag
    if ck is not None:
        bad = ckpt_mismatch(kind, ck.get("meta"), cfg, own_tag=(src == tag))
        if bad:
            log.warning(f"  [{kind}] checkpoint {arm} seed{seed} (tag {src}) trained under different knobs {bad} → retrain"); ck = None
    if ck is not None:
        net = make_net(kind, cfg)
        net.load_state_dict(ck["state_dict"]); log.info(f"  [{kind}] loaded checkpoint {arm} seed{seed} (tag {src})")
        return net
    if kind in TD_KINDS:
        t0 = time.time(); net, orc = train_value_td(kind, groups, ttargets, seed, cfg)
        io_utils.save_checkpoint(net, f"{EXP}_{tag}", kind, seed, arm, meta=dict(arm=arm, groups=groups, train_oracle=orc, **cfg.as_dict()))
        log.info(f"  [{kind}] trained {arm} seed{seed}: train oracle {orc} ({time.time()-t0:.0f}s)")
        return net
    if "data" not in cache:
        t0 = time.time(); cache["data"] = collect_value_data(groups, arm_id(arm), band, ttargets, seed, cfg)
        log.info(f"  [value-data] {arm} seed{seed}: {len(cache['data'])} labelled states ({time.time()-t0:.0f}s)")
    data = cache["data"]
    t0 = time.time(); net = train_value_net(kind, data, cfg, seed)
    io_utils.save_checkpoint(net, f"{EXP}_{tag}", kind, seed, arm,
                          meta=dict(arm=arm, groups=groups, pairs=len(cache["data"]), **cfg.as_dict()))
    log.info(f"  [{kind}] trained {arm} seed{seed} ({time.time()-t0:.0f}s)")
    return net


BEAM_POLICIES = ("beam", "beam_vrel", "beam_sketch")
RESULT_KEYS = ("T", "K", "greedy_k", "value_k", "greedy_obj", "train_targets", "eval_targets", "source", "depth", "b", "k_expand", "rollouts",
               "rollout_len", "value_epochs", "value_batch", "lr", "sig_states", "sig_k", "kband", "mask_connected", "value_label", "value_fam",
               "screen_mult", "beam_bands", "sketch_k", "sketch_q_refresh", "sketch_lam_scale", "warm", "noise_k", "pswap_model", "mixb_eps",
               "mixb_rollouts", "mixb_min_count", "td_gamma", "td_bonus", "td_n", "td_eps", "td_iters", "td_rollouts", "td_steps", "oracle_lam",
               "oracle_q", "warm_cache", "target_cache", "families", "arm_weights", "ckpt_tag", "nodes", "uct_expand", "uct_depth", "uct_c", "uct_widen", "uct_widen_c", "pred_dir", "metrics",
               "sa_budget", "sa_tau0", "sa_tau1", "sa_tau_fixed", "pred_lofo", "oracle_apl", "stop_band", "ext_calib", "ext_model", "max_oracle")
P2_DEFAULTS = {"nodes": 200, "uct_expand": 8, "uct_depth": 3, "uct_c": float(np.sqrt(2)), "uct_widen": 0.0, "uct_widen_c": 1.0, "pred_dir": None, "metrics": "full5",
               "sa_budget": None, "sa_tau0": 0.3, "sa_tau1": 0.01, "sa_tau_fixed": 0.05, "pred_lofo": False,
               "oracle_apl": "nx", "stop_band": None, "ext_calib": None, "ext_model": None, "max_oracle": None}
_LEGACY_WARNED = set()


def result_fingerprint(cfg, arm, seed, policy):
    cur = cfg.as_dict()
    ident = {k: _norm_cfg(cur.get(k)) for k in RESULT_KEYS}
    for _k, _v in P2_DEFAULTS.items():
        if _k in ident and ident[_k] == _norm_cfg(_v):
            ident.pop(_k)
    ident.update(arm=str(arm), seed=int(seed), policy=str(policy))
    if cur.get("lam_lofo"):
        ident["sketch_lam_scale"] = {f: round(float(v), 6) for f, v in sorted(cur["lam_lofo"].items())}
    if str(policy) in PRED_POLICIES and cur.get("pred_dir"):
        ident["pred_model"] = (PR.model_digest_lofo if cur.get("pred_lofo") else PR.model_digest)(cur["pred_dir"], str(arm), str(policy))
    if EXT.kind(policy) in ("nnpred", "nnpred_k8", "nnpreds", "nnpreds_k8") and cur.get("ext_model"):
        ident["ext_model_md5"] = EXT.model_digest(cur["ext_model"], str(arm), str(policy))
    if EXT.kind(policy) in ("gbp", "gbp_k8", "gbps", "gbps_k8") and cur.get("pred_dir"):
        _f = PR.model_path(cur["pred_dir"], str(arm), "LS" if str(policy).startswith("gbps") else "L", "top8" if str(policy).endswith("k8") else "top1")
        ident["pred_model"] = hashlib.md5(_f.read_bytes()).hexdigest()[:12] if _f.exists() else "missing"
    if EXT.kind(policy) == "cv" and cur.get("ext_calib"):
        ident["ext_calib_md5"] = hashlib.md5(Path(cur["ext_calib"]).read_bytes()).hexdigest()[:12]
    return ident, hashlib.md5(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]


def lofo_cols(cfg, fam):
    cols = {"load1": round(os.getloadavg()[0], 1)}
    if getattr(cfg, "pred_lofo", False) or getattr(cfg, "lam_lofo", None):
        cols["lofo_excl"] = fam
    return cols


def lofo_model(cfg, arm, pol, fam, cache):
    key = (pol, fam)
    if key not in cache:
        cache[key] = PR.load(str(Path(cfg.pred_dir) / f"excl_{fam}"), arm, PR.FEAT_OF[pol], PR.OBJ_OF[pol])
        log.info(f"  [{pol}] LOFO model excl_{fam}: {cache[key].path.name} ({cache[key].meta.get('selected')}, hash {cache[key].hash})")
    return cache[key]


def _save_graph(od, arm, seed, pol, ti, hit_edges, final_edges):
    gd = od / "graphs"; gd.mkdir(exist_ok=True)
    np.savez_compressed(gd / f"{arm}_seed{seed}_{pol}_t{ti}.npz",
                        hit=np.asarray(hit_edges if hit_edges else np.zeros((0, 2)), dtype=np.int32).reshape(-1, 2),
                        final=np.asarray(final_edges if final_edges else np.zeros((0, 2)), dtype=np.int32).reshape(-1, 2))


def _graphs_ok(od, arm, seed, pol, n_targets, cfg):
    if not getattr(cfg, "save_graphs", False):
        return True
    for ti in range(n_targets):
        f = od / "graphs" / f"{arm}_seed{seed}_{pol}_t{ti}.npz"
        try:
            with np.load(f) as z:
                _ = z["hit"], z["final"]
        except Exception:
            return False
    return True


def _meta_path(path):
    return path.with_name(path.name + ".meta.json")


def _result_ok(path, cfg, arm, seed, n_targets, policy=None):
    if not path.exists():
        return False
    try:
        d = pd.read_csv(path)
    except Exception:
        return False
    if "arm" not in d or "seed" not in d or not (d["arm"].astype(str) == str(arm)).all() or not (pd.to_numeric(d["seed"], errors="coerce") == int(seed)).all():
        return False
    if policy is None:
        if len(d) != 1:
            return False
    else:
        if "policy" not in d or set(d["policy"].astype(str)) != {policy}:
            return False
        ok, why = RC.rows_ok(d, policy, n_targets, beam_bands=tuple(cfg.beam_bands), bands=bands_of(cfg))
        if not ok:
            log.warning(f"  [reuse-guard] {path.name}: {why}")
            return False
    mp = _meta_path(path)
    if mp.exists():
        try:
            saved = json.loads(mp.read_text()).get("fingerprint")
        except Exception:
            return False
        return saved == result_fingerprint(cfg, arm, seed, policy or "signal")[1]
    if str(path) not in _LEGACY_WARNED:
        _LEGACY_WARNED.add(str(path)); log.warning(f"  [reuse-guard] {path.name}: pre-fingerprint result — accepted on content checks only")
    return True


def bands_of(cfg):
    sb = getattr(cfg, "stop_band", None)
    if not sb:
        return BANDS
    if sb not in BANDS:
        raise ValueError(f"stop_band {sb!r} not in BANDS")
    return [b for b in BANDS if float(b[1:]) >= float(sb[1:])]


def _set_aside(path):
    if path.exists():
        dst = path.with_name(f".{path.name}.incomplete.{time.strftime('%Y%m%d%H%M%S')}.{os.getpid()}")
        os.replace(path, dst); log.warning(f"  [reuse-guard] {path.name} incomplete → moved to {dst.name}, recomputing")
        mp = _meta_path(path)
        if mp.exists():
            os.replace(mp, mp.with_name(f".{mp.name}.incomplete.{time.strftime('%Y%m%d%H%M%S')}.{os.getpid()}"))


def _to_csv_atomic(df, path, fingerprint=None):
    if fingerprint is not None:
        mp = _meta_path(path); tm = mp.with_name(f".{mp.name}.{os.uname().nodename}.{os.getpid()}.tmp")
        tm.write_text(json.dumps(dict(fingerprint=fingerprint[1], ident=fingerprint[0], git=io_utils.git_commit(), time=time.strftime("%Y-%m-%d %H:%M:%S")), default=str))
        os.replace(tm, mp)
    tmp = path.with_name(f".{path.name}.{os.uname().nodename}.{os.getpid()}.tmp"); df.to_csv(tmp, index=False); os.replace(tmp, path)


def _job_manifest(tag, arm, seed, policies, cfg):
    import socket
    rec = dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), git=io_utils.git_commit(), host=socket.gethostname(), pid=os.getpid(), arm=arm,
               seed=seed, policies=list(policies), cfg=cfg.as_dict())
    try:
        with (io_utils.MANI / f"{EXP}_{tag}__{arm}_seed{seed}.jsonl").open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")
    except Exception as e:
        log.warning(f"  [manifest] write failed: {e}")


def arm_weights_of(arm):
    w = {g: w_ for g, w_ in P.arm_spec(arm)}
    return None if len(set(w.values())) == 1 else w


def run_job(arm, seed, policies, tag, cfg):
    set_nodes(getattr(cfg, "nodes", 200))
    set_metrics(getattr(cfg, "metrics", "full5"))
    set_oracle_lam(getattr(cfg, "oracle_lam", "eigsh"))
    set_oracle_q(getattr(cfg, "oracle_q", "cnm"))
    set_oracle_apl(getattr(cfg, "oracle_apl", "nx"))
    cfg.ext_arm = arm
    od = out_dir(tag); band = get_band(TRAIN_BAND); groups = P.arm_groups(arm)
    weights = {g: w for g, w in P.arm_spec(arm)}; weights = None if len(set(weights.values())) == 1 else weights
    cfg.arm_weights = weights
    n_t = cfg.eval_targets
    todo = [p for p in policies if not (_result_ok(od / f"{arm}_seed{seed}_{p}.csv", cfg, arm, seed, n_t, p)
                                        and _graphs_ok(od, arm, seed, p, n_t, cfg))]
    done_marker = od / f"{arm}_seed{seed}.DONE"
    if not todo:
        done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S")); log.info(f"[{arm} seed{seed}] all done — skip"); return
    log.info(f"[{arm} seed{seed}] groups={groups} todo={todo}")
    ttargets, etargets = targets_for(cfg, seed)
    n_t = len(etargets)
    todo = [p for p in policies if not (_result_ok(od / f"{arm}_seed{seed}_{p}.csv", cfg, arm, seed, n_t, p) and _graphs_ok(od, arm, seed, p, n_t, cfg))]
    for p_ in todo:
        _set_aside(od / f"{arm}_seed{seed}_{p_}.csv")
    _job_manifest(tag, arm, seed, policies, cfg)
    ttargets_only = [t for t, _ in ttargets]
    eenvs = []; setup_s = []
    for ti, (t, _) in enumerate(etargets):
        t_s = time.perf_counter(); env = count_env(make_eval_env(t, cfg)); reps_ = reset_connected(env, 7000 + seed + ti)[2]
        eenvs.append(env); setup_s.append((round(time.perf_counter() - t_s, 3), reps_))
    t0 = time.time()
    nrep = sum(int(r > 0) for _, r in setup_s)
    log.info(f"  [warm-start] {len(eenvs)} eval targets cached, {nrep} repaired to connected ({time.time()-t0:.0f}s)")
    cache = {}
    cfg.need_sketch = any(p_ in policies for p_ in ("value_sketch", "value_sketch_min"))
    for pol in todo:
        t0 = time.time(); model = None
        if pol in VALUE_KINDS:
            model = get_or_train(pol, arm, groups, seed, tag, cfg, ttargets, band, cache)
        elif pol in MIXB_POLICIES:
            tb_path = od / f"{arm}_seed{seed}_mixb_table.csv"
            if "mixb" not in cache and tb_path.exists():
                cache["mixb"] = load_mixb_table(tb_path); log.info(f"  [mixb] reloaded table {tb_path.name}")
            if "mixb" not in cache:
                t0m = time.time(); cache["mixb"] = train_mixb(groups, ttargets, seed, cfg); save_mixb_table(cache["mixb"], tb_path, cfg)
                log.info(f"  [mixb] trained {arm} seed{seed}: {int(cache['mixb']['cnt'].sum())} steps, train oracle {cache['mixb']['train_oracle']} ({time.time()-t0m:.0f}s)")
            model = cache["mixb"]
        elif fixm_of(pol) is not None:
            mixb_weights(groups, fixm_of(pol), cfg.K)
        elif pol == "beam_vrel":
            model = get_or_train("value_rel", arm, groups, seed, tag, cfg, ttargets, band, cache)
        elif pol in PRED_POLICIES and getattr(cfg, "pred_lofo", False):
            model = None; assert all((Path(cfg.pred_dir) / f"excl_{f}").is_dir() for f in PT.FAMILIES), f"--pred-lofo needs {cfg.pred_dir}/excl_<family>/ for all 7 families"
        elif EXT.is_ext(pol):
            model = EXT.load_model(pol, cfg, arm)
            if EXT.kind(pol) == "cv":
                EXT.cv_tau(cfg, float(pol[2:]))
        elif pol in PRED_POLICIES:
            model = PR.load(cfg.pred_dir, arm, PR.FEAT_OF[pol], PR.OBJ_OF[pol])
            cfg.pred_hash = (getattr(cfg, "pred_hash", "") + f"{pol}:{model.hash};") if f"{pol}:" not in getattr(cfg, "pred_hash", "") else cfg.pred_hash
            log.info(f"  [{pol}] model {model.path.name} ({model.meta.get('selected')}, hash {model.hash})")
        rows = []; lofo_cache = {}
        for ti, (t, fam) in enumerate(etargets):
            env = eenvs[ti]; sd = 7000 + seed + ti
            if getattr(cfg, "lam_lofo", None):
                cfg.sketch_lam_scale = float(cfg.lam_lofo[fam])
            if pol in PRED_POLICIES and getattr(cfg, "pred_lofo", False):
                model = lofo_model(cfg, arm, pol, fam, lofo_cache)
            info = env.info.get(sd, {}) if isinstance(env, NoisedEnv) else {}
            if pol in ("beam", "beam_vrel", "beam_sketch"):
                res0 = None
                sh = graph_hash(reset_connected(env, sd)[0])
                for b in bands_of(cfg):
                    acc_b = {}; n_ev0 = getattr(env, "n_eval", 0); t_w0 = time.perf_counter(); t_ev0 = getattr(env, "t_eval", 0.0)
                    if b in cfg.beam_bands:
                        if pol == "beam_sketch":
                            sat, orc = beam_sketch_rollout(env, groups, sd, cfg, get_band(b), fam=fam, acc=acc_b)
                        else:
                            sat, orc = beam_rollout(env, groups, sd, cfg, get_band(b), net=model if pol == "beam_vrel" else None, fam=fam, acc=acc_b)
                    else:
                        sat, orc = np.nan, 0
                    tot = getattr(env, "n_eval", 0) - n_ev0; tw = round(time.perf_counter() - t_w0, 3)
                    if cfg.save_graphs and b == SAVE_BAND and b in cfg.beam_bands:
                        _save_graph(od, arm, seed, pol, ti, LAST_HIT_GRAPH["edges"], LAST_HIT_GRAPH["edges"])
                    row = {**io_utils.meta_cols(EXP, tag, seed, arm=arm, policy=pol, target=ti, family=fam, band=b),
                           "sat": sat, "oracle_calls": orc, "oracle_to_hit": orc, "t_hit": np.nan, "steps": cfg.T, "start_hash": sh,
                           "oracle_total": tot, "oracle_total_to_hit": tot, "t_wall": tw, "t_to_hit": tw, "t_setup": setup_s[ti][0],
                           "oracle_time": round(getattr(env, "t_eval", 0.0) - t_ev0, 3), **lofo_cols(cfg, fam)}
                    if pol == "beam_sketch":
                        row.update(sketch_calls=acc_b.get("sketch_calls", 0), sketch_cand=acc_b.get("sketch_cand", 0),
                                   sketch_time=acc_b.get("sketch_time", 0.0), sketch_lam_scale=cfg.sketch_lam_scale)
                    if getattr(cfg, "max_oracle", None) is not None:
                        row.update({k: acc_b[k] for k in ("beam_best_worst", "beam_depth_hit", "beam_depth") if k in acc_b})
                    rows.append({**row, **info})
                continue
            out = sa_rollout(pol, env, groups, sd, cfg, fam=fam, weights=weights) if pol in SA_POLICIES else rollout(pol, model, env, groups, sd, cfg, fam=fam, weights=weights)
            if cfg.save_graphs:
                _save_graph(od, arm, seed, pol, ti, out.get("g_hit"), out.get("g_final"))
            for b in bands_of(cfg):
                bd = get_band(b); hit = next((i for i, r_ in enumerate(out["traj"]) if TB.sat_band(r_, bd)), None)
                rows.append({**io_utils.meta_cols(EXP, tag, seed, arm=arm, policy=pol, target=ti, family=fam, band=b),
                             "sat": int(TB.sat_traj(out["traj"], get_band(b))),
                             "t_hit": hit if hit is not None else np.nan, "oracle_to_hit": out["oracle_cum"][hit] if hit is not None else out["oracle"],
                             "oracle_total": out["oracle_total"], "oracle_total_to_hit": out["tot_cum"][hit] if hit is not None else out["oracle_total"],
                             "t_wall": out["t_wall"], "t_to_hit": round(out["wall_cum"][hit], 3) if hit is not None else out["t_wall"], "t_setup": setup_s[ti][0],
                             "oracle_time": out.get("oracle_time", np.nan), **lofo_cols(cfg, fam),
                             "kbar_start": round(out["kbar_start"], 3), "kbar_final": round(out["kbar_final"], 3),
                             "best_worst": round(out["best_worst"], 4), "oracle_calls": out["oracle"], "steps": out["steps"],
                             "k_eff": out["k_eff"], "fallback_rate": out["fallback_rate"], "mask_rate": out["mask_rate"],
                             "bind0": out["bind0"], "w0": round(float(worst(out["traj"][0])), 4), "start_hash": out["start_hash"], "ws_repairs": out["ws_repairs"],
                             "screen_calls": out["screen_calls"], "pool_overlap": out["pool_overlap"],
                             "sketch_calls": out["sketch_calls"], "sketch_cand": out["sketch_cand"], "sketch_time": out["sketch_time"],
                             "sketch_lam_scale": cfg.sketch_lam_scale,
                             **({"pred_hash": model.hash, "pred_model": model.meta.get("selected", "")} if pol in PRED_POLICIES else {}),
                             **{k: v for k, v in out.items() if k.startswith("bind_frac_")},
                             **{k: v for k, v in out.items() if k.startswith("use_") or k.startswith("grp_")}, **info,
                             **{k: v for k, v in out.items() if k.startswith("sa_")},
                             **{k: v for k, v in out.items() if k.startswith("uct_d")},
                             **{k: v for k, v in out.items() if k.startswith("ext_")},
                             **({k: v for k, v in out.items() if k.startswith("mixb_")} if (pol in MIXB_POLICIES or fixm_of(pol) is not None) else {}),
                             **({"mixb_train_oracle": model["train_oracle"]} if pol in MIXB_POLICIES else {})})
        _to_csv_atomic(pd.DataFrame(rows), od / f"{arm}_seed{seed}_{pol}.csv", fingerprint=result_fingerprint(cfg, arm, seed, pol))
        s = pd.DataFrame(rows); m = {b: round(100 * s[s.band == b].sat.mean(), 1) for b in ("u1.0", "u0.4", "u0.3")}
        extra = "" if pol in ("beam", "beam_vrel", "beam_sketch") else f" kbar={s.kbar_final.mean():.3f} k_eff={s.k_eff.mean():.1f} fallback={s.fallback_rate.mean():.3f}"
        log.info(f"  [{pol}] {arm} seed{seed}: sat={m}{extra} oracle={s.oracle_calls.mean():.0f} ({time.time()-t0:.0f}s)")
    if all(_result_ok(od / f"{arm}_seed{seed}_{p}.csv", cfg, arm, seed, n_t, p) and _graphs_ok(od, arm, seed, p, n_t, cfg) for p in policies) and (cfg.no_signal or _result_ok(od / f"{arm}_seed{seed}_signal.csv", cfg, arm, seed, n_t)):
        done_marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str, nargs="+", default=STAGE1_ARMS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--policies", type=str, nargs="+", default=POLICIES)
    ap.add_argument("--tag", type=str, default="b1")
    ap.add_argument("--source", type=str, default="mix7")
    ap.add_argument("--train-targets", type=int, default=14); ap.add_argument("--eval-targets", type=int, default=28)
    ap.add_argument("--metrics", type=str, default="full5", choices=sorted(METRIC_SETS))
    ap.add_argument("--uct-expand", type=int, default=8)
    ap.add_argument("--uct-depth", type=int, default=3); ap.add_argument("--uct-c", type=float, default=float(np.sqrt(2)))
    ap.add_argument("--uct-widen", type=str, default="0")
    ap.add_argument("--pred-lofo", action="store_true")
    ap.add_argument("--pred-dir", type=str, default=None)
    ap.add_argument("--uct-widen-c", type=float, default=1.0)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--save-graphs", action="store_true")
    ap.add_argument("--sa-budget", type=int, default=None)
    ap.add_argument("--sa-tau0", type=float, default=0.3); ap.add_argument("--sa-tau1", type=float, default=0.01); ap.add_argument("--sa-tau-fixed", type=float, default=0.05)
    ap.add_argument("--T", type=int, default=60); ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--greedy-k", type=int, default=8); ap.add_argument("--value-k", type=int, default=8)
    ap.add_argument("--value-epochs", type=int, default=1200); ap.add_argument("--value-batch", type=int, default=1)
    ap.add_argument("--kband", type=float, nargs=2, default=list(P.KBAND))
    ap.add_argument("--value-label", type=str, default="clip3", choices=["clip3", "noclip", "relative"])
    ap.add_argument("--value-fam", action="store_true")
    ap.add_argument("--value-rollouts", type=int, default=2); ap.add_argument("--value-rollout-len", type=int, default=10)
    ap.add_argument("--screen-mult", type=int, default=4); ap.add_argument("--beam-bands", type=str, nargs="+", default=["u0.4", "u0.3"])
    ap.add_argument("--sketch-k", type=int, default=8); ap.add_argument("--sketch-q-refresh", type=int, default=1)
    ap.add_argument("--sketch-lam-scale", type=str, default="1.0")
    ap.add_argument("--ckpt-tag", type=str, default=None)
    ap.add_argument("--warm", type=str, default="ws")
    ap.add_argument("--oracle-lam", type=str, default="exact", choices=["eigsh", "eigh", "exact"])
    ap.add_argument("--oracle-q", type=str, default="cnm", choices=["cnm", "best"])
    ap.add_argument("--oracle-apl", type=str, default="nx", choices=["nx", "igraph"])
    ap.add_argument("--max-oracle", type=int, default=None)
    ap.add_argument("--stop-band", type=str, default=None)
    ap.add_argument("--ext-calib", type=str, default=None)
    ap.add_argument("--ext-model", type=str, default=None)
    ap.add_argument("--warm-cache", type=str, default=None)
    ap.add_argument("--target-cache", type=str, default=None)
    ap.add_argument("--families", type=str, nargs="+", default=None)
    a = ap.parse_args()
    warm, noise_k = parse_warm(a.warm)
    _uw = resolve_uct_widen(a.uct_widen, a.uct_widen_c)
    lofo_tag = str(a.sketch_lam_scale).split(":", 2)[2] if str(a.sketch_lam_scale).lower().startswith("auto:lofo:") else None
    cfg = Cfg(pred_lofo=a.pred_lofo, T=a.T, K=a.K, greedy_k=a.greedy_k, value_k=a.value_k, source=a.source,
              train_targets=a.train_targets, eval_targets=a.eval_targets, value_epochs=a.value_epochs,
              value_batch=a.value_batch, kband=tuple(a.kband),
              value_label=a.value_label, value_fam=a.value_fam, rollouts=a.value_rollouts, rollout_len=a.value_rollout_len,
              screen_mult=a.screen_mult, beam_bands=a.beam_bands, warm=warm, noise_k=noise_k, no_signal=True,
              oracle_lam=a.oracle_lam, oracle_q=a.oracle_q, warm_cache=a.warm_cache, target_cache=a.target_cache, families=a.families, ckpt_tag=a.ckpt_tag,
              sketch_k=a.sketch_k, sketch_q_refresh=a.sketch_q_refresh, sketch_lam_scale=(1.0 if lofo_tag else resolve_lam_scale(a.sketch_lam_scale, nodes=a.nodes)), nodes=a.nodes,
              uct_expand=a.uct_expand, uct_depth=a.uct_depth, uct_c=a.uct_c, uct_widen=_uw[0], uct_widen_c=_uw[1], pred_dir=a.pred_dir, metrics=a.metrics,
              sa_budget=a.sa_budget, sa_tau0=a.sa_tau0, sa_tau1=a.sa_tau1, sa_tau_fixed=a.sa_tau_fixed, save_graphs=a.save_graphs,
              oracle_apl=a.oracle_apl, stop_band=a.stop_band, ext_calib=a.ext_calib, ext_model=a.ext_model,
              max_oracle=a.max_oracle)
    for A in a.arms:
        P.arm_groups(A)
    io_utils.save_manifest(EXP, a.tag, dict(arms=a.arms, seeds=a.seeds, policies=a.policies, **cfg.as_dict()))
    (out_dir(a.tag) / "config.json").write_text(json.dumps(dict(arms=a.arms, **cfg.as_dict()), indent=2))
    grid = [(A, s) for s in a.seeds for A in a.arms]
    log.info(f"grid={grid} policies={a.policies} device={DEV}")
    for A, s in grid:
        if lofo_tag:
            cfg.lam_lofo = {f: resolve_lam_scale(f"auto:{lofo_tag}_excl_{f}", arm=A, nodes=a.nodes) for f in PT.FAMILIES}
            cfg.sketch_lam_scale = float(cfg.lam_lofo[PT.FAMILIES[0]])
            log.info(f"[sketch] LOFO κ {A}: " + ", ".join(f"{f}={v:.3f}" for f, v in cfg.lam_lofo.items()))
        else:
            cfg.sketch_lam_scale = resolve_lam_scale(a.sketch_lam_scale, arm=A, nodes=a.nodes)
        run_job(A, s, a.policies, a.tag, cfg)
    log.info("done")


if __name__ == "__main__":
    main()
