import argparse, pickle, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO / "stv"))
import numpy as np, pandas as pd, networkx as nx
import targets as PT, graph_env as base
KEYS = list(base.KEYS); TOL = np.array([base.TOL[k] for k in KEYS]); BANDS = (0.4, 0.3, 0.2); KS = (1, 2, 4, 8, 16, 32, 64, 128, 200, 256)


def W(metrics, target):
    return float(np.max(np.abs(np.array([metrics[k] - target[k] for k in KEYS]) / TOL)))


def score(res, n_min=PT.MIN_N):
    rows = []
    for r in res:
        Ws = np.array([W(s["metrics"], r["t"]) if (s["metrics"] and s["n"] >= n_min) else np.inf for s in r["samples"]])
        for k in KS:
            if k > len(Ws):
                continue
            b = float(Ws[:k].min())
            rows.append(dict(seed=r["seed"], target=r["target"], family=r["family"], k=k, W_best=b, valid_frac=float(np.isfinite(Ws[:k]).mean()),
                             **{f"sat_u{u}": float(b <= u) for u in BANDS}))
    return pd.DataFrame(rows)


def selftest():
    import run_experiment as X
    X.set_oracle_lam("exact"); rng = np.random.default_rng(5); bad = 0
    for i in range(20):
        t = PT.sample_one(PT.FAMILIES[i % 7], rng); G = nx.Graph(); G.add_nodes_from(range(t["_n_src"])); G.add_edges_from(t["_graph_edges"])
        tgt = {k: t[k] + rng.normal(0, 0.02) for k in KEYS}
        env = base.SwapEnv.__new__(base.SwapEnv); env.targets = tgt
        w_env = float(np.abs(base.SwapEnv.residuals(env, G)).max()); w_here = W(PT.target_from(G), tgt)
        bad += abs(w_env - w_here) > 1e-3
    if bad:
        sys.exit(f"selftest FAIL: {bad} / 20 graphs where W differs from SwapEnv.residuals by > 1e-3")
    print(f"selftest: {bad} / 20 graphs where W differs from SwapEnv.residuals by > 1e-3 (lambda2 eigsh-vs-dense and 6-dp rounding allowed)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--samples"); ap.add_argument("--out"); ap.add_argument("--selftest", action="store_true"); a = ap.parse_args()
    if a.selftest:
        sys.path.insert(0, str(REPO / "experiments")); selftest(); sys.exit()
    res = pickle.load(open(a.samples, "rb")); res = res["results"] if isinstance(res, dict) else res
    df = score(res); df.to_csv(a.out, index=False)
    s = df.groupby(["k", "seed"])[[f"sat_u{u}" for u in BANDS]].mean().groupby("k").agg(["mean", "sem"]) * 100
    print(s.round(1).to_string()); print(f"valid fraction (k={df.k.max()}):", round(df[df.k == df.k.max()].valid_frac.mean(), 3))
