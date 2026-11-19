"""Start graphs of the real-network runs at four randomization strengths: start score, binding statistic, distance to the network, and whether each reversing swap lies in the typed proposal support."""
import argparse, json, math, os, random, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd, networkx as nx

import operators as P, real_networks as REAL, graph_env as base, io_utils

RES = io_utils.RES; S0 = RES / "real_strengths_starts"
STRENGTHS = (0.1, 0.3, 1.0, 3.0)
X = None


def oracle():
    global X
    if X is None:
        import run_experiment as X_
        X_.set_oracle_lam("exact"); X_.set_oracle_apl("igraph"); X_.set_metrics("full5"); X = X_
    return X


def k_of(s, m):
    return int(math.ceil(s * m - 1e-9))


def residuals(G, t):
    m = base.compute_metrics(G, base.KEYS)
    return np.array([(float(m[k]) - t[k]) / base.TOL[k] for k in base.KEYS], np.float32)


def in_support(G, op_rev, res):
    wants = {k: P._want(res, P.KIND_METRIC[k]) for k in P.TSWAP_KINDS}
    active = [k for k in P.TSWAP_KINDS if wants[k] != 0]
    if not active:
        return True, "free"
    n = G.number_of_nodes(); pe = base.laplacian_pe(G, n, 6); ctx = {"fied": pe[:, 0], "stats": P.degree_stats(G), "comm": P.community_labels(G)}
    ok = [k for k in active if P._kind_score(G, op_rev, k, ctx) * wants[k] > 0]
    return bool(ok), ",".join(ok)


def cmd_s0(net, seeds=range(10), support_seeds=(0, 1), n_pos=50):
    X_ = oracle(); t = REAL.real_target(net); n = int(t["_n"]); E0 = t["_graph_edges"]; m = len(E0)
    G_real = nx.Graph(); G_real.add_nodes_from(range(n)); G_real.add_edges_from(E0)
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); residuals(G_real, t); ts.append(time.perf_counter() - t0)
    to = dict(net=net, n=n, m=m, t_o_median=float(np.median(ts)), t_o_all=ts, load1=os.getloadavg()[0])
    ks = {s: k_of(s, m) for s in STRENGTHS}; rows = []; sup = []
    fh = {}
    for sd in seeds:
        f = RES / f"run_real_networks__{net}_greedy_K32" / f"S_tswap_seed{sd}_greedy.csv"
        if f.exists():
            fh[sd] = str(pd.read_csv(f).start_hash.iloc[0])
    for sd in seeds:
        G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from(E0); E0s = P.edge_set(G)
        rng = random.Random(7919 * (7000 + sd) + 17); ops = []; done = 0
        for s in STRENGTHS:
            got = P.noise_swaps(G, ks[s] - done, rng); ops += got; done += len(got)
            res = residuals(G, t); W = float(np.abs(res).max())
            h = X_.graph_hash(G)
            rows.append(dict(net=net, seed=sd, s=s, k=ks[s], noise_done=done, W0=W, bind0=base.KEYS[int(np.argmax(np.abs(res)))],
                             in_u04=bool(W <= 0.4), in_u03=bool(W <= 0.3), start_edge_dist=round(1.0 - len(P.edge_set(G) & E0s) / max(len(E0s), 1), 4),
                             start_hash=h, fixed_horizon_start_hash=fh.get(sd, "") if s == 3.0 else "",
                             hash_eq_fixed_horizon=(h == fh[sd]) if (s == 3.0 and sd in fh) else None,
                             **{f"res_{k}": float(r) for k, r in zip(base.KEYS, res)}))
        if sd in support_seeds:
            for s in STRENGTHS:
                k = ks[s]; pos = sorted(set(np.linspace(1, k, min(n_pos, k)).round().astype(int).tolist()))
                H = nx.Graph(); H.add_nodes_from(range(n)); H.add_edges_from(E0); j = 0
                for jp in pos:
                    while j < jp:
                        P.apply_op(H, ops[j]); j += 1
                    rev = P.swap_reverse(ops[jp - 1]); res = residuals(H, t)
                    ins, kinds = in_support(H, rev, res)
                    sup.append(dict(net=net, seed=sd, s=s, k=k, j=jp, in_support=ins, kinds=kinds, W=float(np.abs(res).max())))
    S0.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(S0 / f"{net}_starts.csv", index=False); pd.DataFrame(sup).to_csv(S0 / f"{net}_support.csv", index=False)
    (S0 / f"{net}_to.json").write_text(json.dumps(to, default=float)); (S0 / f"{net}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    r = pd.DataFrame(rows); sp = pd.DataFrame(sup)
    print(f"[starts] {net}: hash = fixed-horizon start {int(r.hash_eq_fixed_horizon.fillna(False).sum())}/{int(r.hash_eq_fixed_horizon.notna().sum())}; in-band u0.4 by s "
          f"{r.groupby('s').in_u04.mean().round(2).to_dict()}; support by s {sp.groupby('s').in_support.mean().round(2).to_dict() if len(sp) else {}}; "
          f"t_o {to['t_o_median']:.4f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["starts"]); ap.add_argument("--net")
    ap.add_argument("--n-pos", type=int, default=50); ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    cmd_s0(a.net, seeds=range(a.seeds), n_pos=a.n_pos)


if __name__ == "__main__":
    main()
