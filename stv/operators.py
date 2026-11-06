"""Edit operators (swap, 3-swap, rewire, add/delete, joint-degree swap) and typed, screened and diverse proposals."""
import random
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

import graph_env as base

OP_TYPES = ["swap", "jdm", "swap3", "rewire", "add", "del", "compound"]
OP_INDEX = {t: i for i, t in enumerate(OP_TYPES)}
BASE_OPS = ["swap", "jdm", "swap3", "rewire", "adddel"]
TYPED_GROUPS = {f"t{o}": o for o in BASE_OPS}
SCREEN_GROUPS = {f"o{o}": o for o in BASE_OPS}
GROUPS = BASE_OPS + list(TYPED_GROUPS) + list(SCREEN_GROUPS) + ["compound", "pswap"]
ARMS = {
    "S_swap": ["swap"], "S_tswap": ["tswap"], "S_jdm": ["jdm"], "S_swap3": ["swap3"],
    "S_rewire": ["rewire"], "S_adddel": ["adddel"], "S_tadddel": ["tadddel"],
    "S_tjdm": ["tjdm"], "S_tswap3": ["tswap3"], "S_trewire": ["trewire"],
    "S_oswap": ["oswap"], "S_ojdm": ["ojdm"], "S_oswap3": ["oswap3"], "S_orewire": ["orewire"], "S_oadddel": ["oadddel"],
    "S_pswap": ["pswap"],
    "L0": ["swap"], "L1": ["swap", "tswap"], "L2": ["swap", "tswap", "rewire"],
    "L3": ["swap", "tswap", "rewire", "adddel"], "L4": ["swap", "tswap", "rewire", "adddel", "compound"],
}
KBAND = (5.5, 6.5)
AF_DIM = len(OP_TYPES) + 4 + 4 + 2
EXT_DIM = AF_DIM + 4
KEYS = base.KEYS
IDX = {k: KEYS.index(k) for k in KEYS}
TSWAP_KINDS = ["r", "tri", "comm", "spec"]
TADDDEL_KINDS = ["spec", "tri"]
_TRIES = 60


DIVERSE_GROUPS = {f"d{o}": o for o in BASE_OPS}
GROUPS += list(DIVERSE_GROUPS)


def arm_groups(name):
    return [g for g, _ in arm_spec(name)]


def arm_spec(name):
    if name in ARMS:
        return [(g, 1) for g in ARMS[name]]
    out = []
    for part in name.split("+"):
        g, _, w = part.partition("@")
        if g not in GROUPS:
            raise ValueError(f"unknown group '{g}' in arm '{name}'")
        out.append((g, int(w) if w else 1))
    if len({g for g, _ in out}) != len(out) or any(w <= 0 for _, w in out):
        raise ValueError(f"arm '{name}': each group at most once with a positive weight (use g@w for shares)")
    return out


def apply_op(G, op):
    t, p = op[0], op[1]
    if t in ("swap", "jdm"):
        a, b, c, d = p; G.remove_edge(a, b); G.remove_edge(c, d); G.add_edge(a, d); G.add_edge(c, b)
    elif t == "swap3":
        a, b, c, d, e, f = p
        G.remove_edge(a, b); G.remove_edge(c, d); G.remove_edge(e, f); G.add_edge(a, d); G.add_edge(c, f); G.add_edge(e, b)
    elif t == "rewire":
        u, v, w = p; G.remove_edge(u, v); G.add_edge(u, w)
    elif t == "add":
        G.add_edge(p[0], p[1])
    elif t == "del":
        G.remove_edge(p[0], p[1])
    elif t == "compound":
        for sub in p:
            apply_op(G, sub)
    else:
        raise ValueError(t)


def revert_op(G, op):
    t, p = op[0], op[1]
    if t in ("swap", "jdm"):
        a, b, c, d = p; G.remove_edge(a, d); G.remove_edge(c, b); G.add_edge(a, b); G.add_edge(c, d)
    elif t == "swap3":
        a, b, c, d, e, f = p
        G.remove_edge(a, d); G.remove_edge(c, f); G.remove_edge(e, b); G.add_edge(a, b); G.add_edge(c, d); G.add_edge(e, f)
    elif t == "rewire":
        u, v, w = p; G.remove_edge(u, w); G.add_edge(u, v)
    elif t == "add":
        G.remove_edge(p[0], p[1])
    elif t == "del":
        G.add_edge(p[0], p[1])
    elif t == "compound":
        for sub in reversed(p):
            revert_op(G, sub)
    else:
        raise ValueError(t)


def edge_lists(op):
    t, p = op[0], op[1]
    if t == "compound":
        rem, add = [], []
        for s in p:
            r, a = edge_lists(s); rem += r; add += a
        return rem, add
    if t in ("swap", "jdm"):
        a, b, c, d = p; return [(a, b), (c, d)], [(a, d), (c, b)]
    if t == "swap3":
        a, b, c, d, e, f = p; return [(a, b), (c, d), (e, f)], [(a, d), (c, f), (e, b)]
    if t == "rewire":
        u, v, w = p; return [(u, v)], [(u, w)]
    if t == "add":
        return [], [(p[0], p[1])]
    return [(p[0], p[1])], []


def op_nodes(op):
    if op[0] == "compound":
        s = set()
        for sub in op[1]:
            s |= op_nodes(sub)
        return s
    return set(op[1])


def op_group(op):
    return op[2] if len(op) > 2 else op[0]


def kbar(G):
    return 2.0 * G.number_of_edges() / G.number_of_nodes()


def _valid(G, op):
    t, p = op[0], op[1]
    if t in ("swap", "jdm"):
        a, b, c, d = p
        return (len({a, b, c, d}) == 4 and G.has_edge(a, b) and G.has_edge(c, d)
                and not G.has_edge(a, d) and not G.has_edge(c, b))
    if t == "swap3":
        a, b, c, d, e, f = p
        return (len({a, b, c, d, e, f}) == 6 and G.has_edge(a, b) and G.has_edge(c, d) and G.has_edge(e, f)
                and not G.has_edge(a, d) and not G.has_edge(c, f) and not G.has_edge(e, b))
    if t == "rewire":
        u, v, w = p
        return (w != u and w != v and G.has_edge(u, v) and not G.has_edge(u, w))
    if t == "add":
        return p[0] != p[1] and not G.has_edge(p[0], p[1])
    if t == "del":
        return G.has_edge(p[0], p[1])
    raise ValueError(t)


def _sample_swap(G, edges, rng, tries=20):
    for _ in range(tries):
        (a, b), (c, d) = rng.choice(edges), rng.choice(edges)
        if rng.random() < 0.5:
            c, d = d, c
        op = ("swap", (a, b, c, d), "swap")
        if _valid(G, op):
            return op
    return None


def _sample_jdm(G, edges, rng, tries=_TRIES):
    deg = G.degree
    for _ in range(tries):
        (a, b), (c, d) = rng.choice(edges), rng.choice(edges)
        if rng.random() < 0.5:
            c, d = d, c
        if deg(b) != deg(d) and deg(a) != deg(c):
            continue
        op = ("jdm", (a, b, c, d), "jdm")
        if _valid(G, op):
            return op
    return None


def _sample_swap3(G, edges, rng, tries=40):
    for _ in range(tries):
        (a, b), (c, d), (e, f) = rng.choice(edges), rng.choice(edges), rng.choice(edges)
        if rng.random() < 0.5:
            c, d = d, c
        if rng.random() < 0.5:
            e, f = f, e
        op = ("swap3", (a, b, c, d, e, f), "swap3")
        if _valid(G, op):
            return op
    return None


def _sample_rewire(G, edges, nodes, rng, tries=20):
    for _ in range(tries):
        u, v = rng.choice(edges)
        if rng.random() < 0.5:
            u, v = v, u
        w = rng.choice(nodes)
        op = ("rewire", (u, v, w), "rewire")
        if _valid(G, op) and G.degree(v) >= 2:
            return op
    return None


def _sample_add(G, nodes, rng, tries=20, group="adddel"):
    for _ in range(tries):
        u, w = rng.choice(nodes), rng.choice(nodes)
        op = ("add", (u, w), group)
        if _valid(G, op):
            return op
    return None


def _sample_del(G, edges, rng, tries=20, group="adddel"):
    for _ in range(tries):
        u, v = rng.choice(edges)
        if G.degree(u) >= 2 and G.degree(v) >= 2:
            return ("del", (u, v), group)
    return None


def guard_dirs(G, kband, m_shift=0):
    n = G.number_of_nodes(); kb = 2.0 * (G.number_of_edges() + m_shift) / n
    return (kb + 2.0 / n <= kband[1]), (kb - 2.0 / n >= kband[0])


def _sample_adddel(G, edges, nodes, rng, kband):
    can_add, can_del = guard_dirs(G, kband)
    choices = [c for c, ok in (("add", can_add), ("del", can_del)) if ok]
    if not choices:
        return None
    c = rng.choice(choices)
    return _sample_add(G, nodes, rng) if c == "add" else _sample_del(G, edges, rng)


def degree_stats(G):
    deg = dict(G.degree())
    S1 = sum(deg[u] * deg[v] for u, v in G.edges())
    S2 = sum(d * d for d in deg.values()); S3 = sum(d ** 3 for d in deg.values())
    return S1, S2, S3, G.number_of_edges()


def assort_from_stats(S1, S2, S3, m):
    if m <= 0:
        return 0.0
    mu = S2 / (2.0 * m); var = S3 / (2.0 * m) - mu * mu
    if var <= 1e-12:
        return 0.0
    return (S1 / m - mu * mu) / var


def _touch_dp(G, nodes):
    deg = G.degree; seen = set(); s = 0
    for u in nodes:
        du = deg(u)
        for v in G.neighbors(u):
            e = (u, v) if u < v else (v, u)
            if e in seen:
                continue
            seen.add(e); s += du * deg(v)
    return s


def _cn(G, x, y):
    return sum(1 for _ in nx.common_neighbors(G, x, y))


def tri_delta(G, op):
    if op[0] == "compound":
        tot = 0; applied = []
        try:
            for sub in op[1]:
                tot += tri_delta(G, sub); apply_op(G, sub); applied.append(sub)
        finally:
            for sub in reversed(applied):
                revert_op(G, sub)
        return tot
    rem, add = edge_lists(op); tot = 0; done = []
    try:
        for u, v in rem:
            tot -= _cn(G, u, v); G.remove_edge(u, v); done.append(("r", u, v))
        for u, v in add:
            tot += _cn(G, u, v); G.add_edge(u, v); done.append(("a", u, v))
    finally:
        for kind, u, v in reversed(done):
            (G.add_edge if kind == "r" else G.remove_edge)(u, v)
    return tot


def local_deltas(G, op, stats):
    S1, S2, S3, m = stats
    nodes = op_nodes(op)
    deg0 = {v: G.degree(v) for v in nodes}; dp0 = _touch_dp(G, nodes); dtri = tri_delta(G, op)
    apply_op(G, op)
    deg1 = {v: G.degree(v) for v in nodes}; dp1 = _touch_dp(G, nodes); m1 = G.number_of_edges()
    revert_op(G, op)
    dS2 = sum(deg1[v] ** 2 - deg0[v] ** 2 for v in nodes); dS3 = sum(deg1[v] ** 3 - deg0[v] ** 3 for v in nodes)
    dS1 = dp1 - dp0
    r0 = assort_from_stats(S1, S2, S3, m); r1 = assort_from_stats(S1 + dS1, S2 + dS2, S3 + dS3, m1)
    ddeg_all = sorted((deg1[v] - deg0[v] for v in nodes), key=lambda x: -abs(x))
    return dict(dm=m1 - m, ddeg=(ddeg_all + [0, 0, 0, 0])[:4], ddeg_all=ddeg_all, dS1=dS1, dr=r1 - r0, dtri=dtri)


def endpoint_degrees(G, op):
    if op[0] == "compound":
        return endpoint_degrees(G, op[1][0])
    ds = [G.degree(v) for v in op[1]]
    return (ds + [0, 0, 0, 0])[:4]


def op_features(G, op, stats):
    f = np.zeros(AF_DIM, np.float32); k = len(OP_TYPES)
    f[OP_INDEX[op[0]]] = 1.0
    f[k:k + 4] = np.asarray(endpoint_degrees(G, op), np.float32) / 10.0
    ld = local_deltas(G, op, stats)
    f[k + 4:k + 8] = np.asarray(ld["ddeg"], np.float32)
    f[k + 8] = 10.0 * ld["dr"]; f[k + 9] = ld["dtri"] / 10.0
    return f, ld


def fiedler_features(op, fied):
    if fied is None:
        return [0.0, 0.0, 0.0]
    rem, add = edge_lists(op)
    fr = float(np.mean([abs(fied[u] - fied[v]) for u, v in rem])) if rem else 0.0
    fa = float(np.mean([abs(fied[u] - fied[v]) for u, v in add])) if add else 0.0
    if op[0] in ("swap", "jdm"):
        a, b, c, d = op[1]; prod = float((fied[a] - fied[b]) * (fied[c] - fied[d]))
    else:
        prod = fr * fa
    return [fr, fa, prod]


def ext_features(G, op, stats, fied, w0):
    f, ld = op_features(G, op, stats)
    return np.concatenate([f, np.asarray(fiedler_features(op, fied) + [w0], np.float32)]), ld


def _cut_delta(op, fied):
    rem, add = edge_lists(op); s = lambda u, v: int((fied[u] >= 0) != (fied[v] >= 0))
    return sum(s(u, v) for u, v in add) - sum(s(u, v) for u, v in rem)


def _intra_delta(op, comm):
    rem, add = edge_lists(op); s = lambda u, v: int(comm[u] == comm[v])
    return sum(s(u, v) for u, v in add) - sum(s(u, v) for u, v in rem)


def community_labels(G):
    lab = {}
    for i, c in enumerate(greedy_modularity_communities(G)):
        for v in c:
            lab[v] = i
    return lab


def _want(res, key):
    r = res[IDX[key]]
    return 0 if abs(r) < 0.5 else (1 if r < 0 else -1)


def _kind_score(G, op, kind, ctx):
    if kind == "r":
        return local_deltas(G, op, ctx["stats"])["dr"]
    if kind == "tri":
        return tri_delta(G, op)
    if kind == "comm":
        return _intra_delta(op, ctx["comm"])
    if kind == "spec":
        return _cut_delta(op, ctx["fied"])
    raise ValueError(kind)


KIND_METRIC = {"r": "assortativity", "tri": "clustering", "comm": "modularity", "spec": "spectral_gap"}


def _uniform_sampler(base_op, G, edges, nodes, rng, kband):
    if base_op == "swap":
        return _sample_swap(G, edges, rng)
    if base_op == "jdm":
        return _sample_jdm(G, edges, rng)
    if base_op == "swap3":
        return _sample_swap3(G, edges, rng)
    if base_op == "rewire":
        return _sample_rewire(G, edges, nodes, rng)
    if base_op == "adddel":
        return _sample_adddel(G, edges, nodes, rng, kband)
    raise ValueError(base_op)


def _sample_typed(base_op, G, edges, nodes, rng, kind, want, ctx, kband):
    grp = f"t{base_op}"
    if want == 0:
        op = _uniform_sampler(base_op, G, edges, nodes, rng, kband)
        return ((op[0], op[1], f"{grp}:free") if op else None)
    for _ in range(_TRIES):
        op = _uniform_sampler(base_op, G, edges, nodes, rng, kband)
        if op is None:
            continue
        if _kind_score(G, op, kind, ctx) * want > 0:
            return (op[0], op[1], f"{grp}:{kind}")
    return None


def _sample_tadddel(G, edges, nodes, rng, kind, want, ctx, kband, n_pool=30):
    can_add, can_del = guard_dirs(G, kband)
    if want == 0:
        op = _sample_adddel(G, edges, nodes, rng, kband)
        return ((op[0], op[1], "tadddel:free") if op else None)
    fied = ctx["fied"]
    key = (lambda u, v: (fied[u] - fied[v]) ** 2) if (kind == "spec" and fied is not None) else (lambda u, v: _cn(G, u, v))
    if want > 0:
        if not can_add:
            return None
        pool = [op for op in (_sample_add(G, nodes, rng, group="tadddel") for _ in range(n_pool)) if op]
        if not pool:
            return None
        op = max(pool, key=lambda o: key(*o[1]))
        return ("add", op[1], f"tadddel:{kind}")
    if not can_del:
        return None
    pool = [op for op in (_sample_del(G, edges, rng, group="tadddel") for _ in range(n_pool)) if op]
    if not pool:
        return None
    op = max(pool, key=lambda o: key(*o[1]))
    return ("del", op[1], f"tadddel:{kind}")


def _sample_compound(G, edges, nodes, rng, kband):
    k = rng.choice([2, 3]); subs = []
    try:
        for _ in range(k):
            kind = rng.choice(["swap", "rewire", "adddel"]); e = list(G.edges())
            s = (_sample_swap(G, e, rng) if kind == "swap" else _sample_rewire(G, e, nodes, rng) if kind == "rewire"
                 else _sample_adddel(G, e, nodes, rng, kband))
            if s is None:
                break
            subs.append(s); apply_op(G, s)
    finally:
        for s in reversed(subs):
            revert_op(G, s)
    if len(subs) < 2:
        return None
    return ("compound", tuple(subs), "compound")


PSWAP_DIM = 16
PSWAP_FEAT_VERSION = "pswap-v1-16"
PSWAP_POOL_MULT = 8


def edge_set(G):
    return {(u, v) if u < v else (v, u) for u, v in G.edges()}


def noise_swaps(G, k, rng, tries_per=200):
    done = []
    for _ in range(int(k)):
        ok = False
        for _t in range(tries_per):
            op = _sample_swap(G, list(G.edges()), rng, tries=1)
            if op is None:
                continue
            apply_op(G, op)
            if nx.is_connected(G):
                done.append(op); ok = True
                break
            revert_op(G, op)
        if not ok:
            break
    return done


def swap_canon(op):
    rem, add = edge_lists(op)
    f = lambda es: frozenset(frozenset(e) for e in es)
    return (f(rem), f(add))


def _pair_stats(G, u, v):
    Nu = set(G[u]); Nv = set(G[v]); cn = Nu & Nv; un = (Nu | Nv) - {u, v}
    return len(cn), (len(cn) / len(un) if un else 0.0), float(sum(1.0 / np.log(G.degree(w)) for w in cn))


def pswap_features(G, op, stats=None):
    a, b, c, d = op[1]
    stats = stats if stats is not None else degree_stats(G)
    ce1, je1, _ = _pair_stats(G, a, b); ce2, je2, _ = _pair_stats(G, c, d)
    cf1, jf1, af1 = _pair_stats(G, a, d); cf2, jf2, af2 = _pair_stats(G, c, b)
    ld = local_deltas(G, op, stats)
    k = {x: G.degree(x) for x in (a, b, c, d)}
    clu = nx.clustering(G, nodes=[a, b, c, d])
    s = lambda x, y: (min(x, y), max(x, y))
    f = (*s(ce1, ce2), *s(cf1, cf2), *s(je1, je2), *s(jf1, jf2), *s(af1, af2), ld["dtri"] / 10.0, 10.0 * ld["dr"],
         *s(np.log1p(abs(k[b] - k[d])), np.log1p(abs(k[a] - k[c]))),
         float(np.mean(sorted(np.log1p(k[x]) for x in (a, b, c, d)))), float(np.mean(sorted(clu[x] for x in (a, b, c, d)))))
    return np.asarray(f, np.float32)


class DtriScorer:
    feat_col = 10

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-np.asarray(X, np.float64)[:, self.feat_col]))
        return np.stack([1.0 - p, p], 1)


def gen_ops(G, groups, K, rng, res=None, kband=KBAND, abl=None, mask_connected=True, weights=None, pswap_model=None):
    if res is None:
        res = np.zeros(5, np.float32)
    w = weights or {gr: 1 for gr in groups}; tot = sum(w[gr] for gr in groups)
    share = {gr: (K * w[gr]) // tot for gr in groups}; share[groups[0]] += K - sum(share.values())
    nodes = list(G.nodes()); edges = list(G.edges()); n = G.number_of_nodes()
    abl = abl or base.Ablation("pc", k_pe=6, frac_bridge=0.5)
    random.seed(rng.randrange(1 << 30))
    pe = base.laplacian_pe(G, n, max(abl.k_pe, 1)); fied = pe[:, 0] if pe.shape[1] > 0 else None
    ctx = {"fied": fied, "stats": None, "comm": None}
    if any(gr in TYPED_GROUPS for gr in groups):
        ctx["stats"] = degree_stats(G); ctx["comm"] = community_labels(G)
    ops = []; st = {"requested": K, "fallback": {}, "kinds": {}}
    for gr in groups:
        cnt = share[gr]; got = []
        if gr == "swap":
            sw = base.sample_candidates_spectral(G, cnt, fied, abl.frac_bridge)
            got = [("swap", tuple(int(x) for x in s), "swap") for s in sw]
        elif gr in TYPED_GROUPS:
            base_op = TYPED_GROUPS[gr]
            kinds = TADDDEL_KINDS if gr == "tadddel" else ([k for k in TSWAP_KINDS if k != "r"] if gr == "tjdm" else TSWAP_KINDS)
            wants = {k: _want(res, KIND_METRIC[k]) for k in kinds}
            active = [k for k in kinds if wants[k] != 0]
            for i in range(cnt):
                if active:
                    kind = active[i % len(active)]; want = wants[kind]
                else:
                    kind, want = kinds[0], 0
                op = (_sample_tadddel(G, edges, nodes, rng, kind, want, ctx, kband) if gr == "tadddel"
                      else _sample_typed(base_op, G, edges, nodes, rng, kind, want, ctx, kband))
                if op is not None:
                    got.append(op); st["kinds"][op[2]] = st["kinds"].get(op[2], 0) + 1
        elif gr == "pswap":
            if pswap_model is None:
                raise ValueError("group 'pswap' is not part of this release")
            pool = []; seen_p = set()
            for _ in range(PSWAP_POOL_MULT * cnt):
                op = _sample_swap(G, edges, rng)
                if op is None:
                    continue
                ck = swap_canon(op)
                if ck not in seen_p:
                    seen_p.add(ck); pool.append(("swap", op[1], "pswap"))
            if pool:
                S_ = degree_stats(G)
                sc = pswap_model.predict_proba(np.stack([pswap_features(G, o, S_) for o in pool]))[:, 1]
                for i in sorted(range(len(pool)), key=lambda i: (-float(sc[i]), i)):
                    if len(got) >= cnt:
                        break
                    if mask_connected:
                        apply_op(G, pool[i]); ok = nx.is_connected(G); revert_op(G, pool[i])
                        if not ok:
                            continue
                    got.append(pool[i])
            st["kinds"]["pswap"] = len(got)
        elif gr in ("oswap", "dswap"):
            sw = base.sample_candidates_spectral(G, cnt, fied, abl.frac_bridge)
            got = [("swap", tuple(int(x) for x in s_), gr) for s_ in sw]
        elif gr in SCREEN_GROUPS or gr in DIVERSE_GROUPS or gr in ("jdm", "swap3", "rewire", "adddel", "compound"):
            base_op = SCREEN_GROUPS.get(gr, DIVERSE_GROUPS.get(gr, gr))
            for _ in range(cnt * 3):
                if len(got) >= cnt:
                    break
                if base_op == "compound":
                    op = _sample_compound(G, edges, nodes, rng, kband)
                else:
                    op = _uniform_sampler(base_op, G, edges, nodes, rng, kband)
                    if op is not None and (gr in SCREEN_GROUPS or gr in DIVERSE_GROUPS):
                        op = (op[0], op[1], gr)
                if op is not None:
                    got.append(op)
        else:
            raise ValueError(gr)
        st["fallback"][gr] = cnt - len(got)
        ops += got
    seen = set(); out = []; masked = 0
    for op in ops:
        key = (op[0], op[1])
        if key in seen:
            continue
        seen.add(key)
        if mask_connected:
            apply_op(G, op); ok = nx.is_connected(G); revert_op(G, op)
            if not ok:
                masked += 1; continue
        out.append(op)
    st["produced"] = len(out); st["masked"] = masked
    return out, fied, st


def op_usage(ops_chosen):
    c = {t: 0 for t in OP_TYPES}; gc = {}
    for op in ops_chosen:
        c[op[0]] += 1; g = op_group(op).split(":")[0]; gc[g] = gc.get(g, 0) + 1
    tot = max(len(ops_chosen), 1)
    out = {f"use_{t}": round(c[t] / tot, 3) for t in OP_TYPES}
    out.update({f"grp_{g}": round(gc.get(g, 0) / tot, 3) for g in GROUPS})
    return out
