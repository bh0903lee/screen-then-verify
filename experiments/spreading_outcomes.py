"""Conditional modularity shifts: SIR calibration and threshold choice on the real network, and spreading outcomes of the edited graphs."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd, networkx as nx

import dynamics as DY, real_networks as REAL, graph_env as base, io_utils

RES = io_utils.RES
OUT_DIR = RES / "modularity_shift"
NETS = ["polbooks", "football", "game_thrones", "fresh_webs__Martins", "webkb__webkb_texas_link1"]
CELLS = ["m10", "m05", "p00", "p05", "p10"]; DQ = {"m10": -0.10, "m05": -0.05, "p00": 0.0, "p05": 0.05, "p10": 0.10}
ARMS = {"H": "full5", "Q": "q1"}
ARM_NAME = {"H": "held", "Q": "modularity_only"}
PHI_GRID = (0.18, 0.25, 0.30, 0.35, 0.40, 0.50)
SIR_TARGETS = (0.10, 0.30, 0.60)
RUNS54 = dict(sir=500, cc=200, phi_real=2000)
SIR_REAL_RUNS = 5000
X = None


def oracle():
    global X
    if X is None:
        import run_experiment as X_
        X_.set_oracle_lam("exact"); X_.set_oracle_apl("igraph"); X_.set_metrics("full5"); X = X_
    return X


def seed_of(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:15], 16)


def real_graph(net):
    t = REAL.real_target(net)
    return t, int(t["_n"]), [(min(u, v), max(u, v)) for u, v in t["_graph_edges"]]


def nx_of(n, edges):
    G = nx.Graph(); G.add_nodes_from(range(n)); G.add_edges_from((int(u), int(v)) for u, v in edges); return G


def five(G, t):
    oracle(); m = base.compute_metrics(G, base.KEYS); m = {k: float(m[k]) for k in base.KEYS}
    r = {k: (m[k] - t[k]) / base.TOL[k] for k in base.KEYS}
    return dict(**{f"m_{k}": v for k, v in m.items()}, W=max(abs(x) for x in r.values()),
                W_local2=max(abs(r["clustering"]), abs(r["assortativity"])),
                W_global3=max(abs(r["avg_path_length"]), abs(r["modularity"]), abs(r["spectral_gap"])))


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp"); tmp.write_text(text); os.replace(tmp, path)


def tag_of(net, arm, cell):
    return f"modularity_shift__{net}_{ARM_NAME[arm]}_dQ{DQ[cell]:+.2f}"


def run_dir(net, arm, cell):
    return RES / f"run_{tag_of(net, arm, cell)}"


def _npz(d, seed, pol="beam_sketch", arm="S_tswap"):
    with np.load(d / "graphs" / f"{arm}_seed{seed}_{pol}_t0.npz") as z:
        return {k: z[k] for k in z.files}


def run_record(net, arm, cell, seed):
    d = run_dir(net, arm, cell); an = "S_tswap"; pol = "beam_sketch"
    f = d / f"{an}_seed{seed}_{pol}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f); r = df[df.band == "u0.4"].iloc[0]
    z = _npz(d, seed, pol, an); hit = z["hit"]
    rec = dict(reached=int(len(hit) > 0), t_wall=float(r.t_wall), start_hash=str(r.start_hash), oracle=int(r.oracle_calls))
    rec.update(stop=str(r.stop_reason), depth=float(r.get("beam_depth_hit", np.nan)) if len(hit) else np.nan)
    rec.update(graph="hit" if len(hit) else "best", edges=hit if len(hit) else z.get("best", np.zeros((0, 2), np.int32)))
    if rec["reached"]:
        rec["stop"] = "hit"
    return rec


OTHER = [k for k in base.KEYS if k != "modularity"]


def drift(five_, t):
    dl = {k: (five_[f"m_{k}"] - t[k]) / base.TOL[k] for k in OTHER}
    return dict(D4=max(abs(v) for v in dl.values()), **{f"delta_{k}": v for k, v in dl.items()})


def cmd_calibrate(net):
    t, n, e0 = real_graph(net); G = nx_of(n, e0); csr = DY.csr_of(G); dd = OUT_DIR / net; dd.mkdir(parents=True, exist_ok=True)
    p = {}
    for tg in SIR_TARGETS:
        r = DY.calibrate_p(csr, tg, rng_seed=seed_of("calib", net, tg), runs=SIR_REAL_RUNS)
        p[f"{tg:.2f}"] = r["p"]; p[f"{tg:.2f}_info"] = r
    cal = dict(net=net, p={k: v for k, v in p.items() if not k.endswith("_info")}, p_info={k: v for k, v in p.items() if k.endswith("_info")})
    atomic_write(dd / "calib.json", json.dumps(cal, indent=1, default=float))
    print(f"[calibrate] {net}: p {cal['p']}", flush=True)


def phi_star(adopt):
    ok = [(abs(a - 0.5), float(p)) for p, a in adopt.items() if 0.2 <= a <= 0.8]
    return min(ok)[1] if ok else None


def cmd_threshold(net):
    t, n, e0 = real_graph(net); G = nx_of(n, e0); csr = DY.csr_of(G); dd = OUT_DIR / net; dd.mkdir(parents=True, exist_ok=True)
    adopt = {f"{phi:.2f}": DY.threshold_summary(csr, phi, "ego", RUNS54["phi_real"], np.random.default_rng(seed_of("phi", net, phi)))["adopt"]
             for phi in PHI_GRID}
    ps = phi_star(adopt); deg = np.diff(csr[0])
    out = dict(net=net, grid=list(PHI_GRID), draws=RUNS54["phi_real"], real_adopt_ego=adopt, phi_star=ps, excluded=ps is None,
               ego_seed_frac=float((1 + deg).mean() / n), real_hash=oracle().graph_hash(G))
    atomic_write(dd / "phi.json", json.dumps(out, indent=1, default=float))
    print(f"[threshold] {net}: phi* = {ps} (adoption {adopt})", flush=True)


def outcomes54(G, t, cal, runs, ghash, phis=PHI_GRID):
    csr = DY.csr_of(G); out = dict(five(G, t))
    for tg in SIR_TARGETS:
        s = DY.sir_summary(csr, cal["p"][f"{tg:.2f}"], runs["sir"], np.random.default_rng(seed_of("sir", ghash, tg)))
        out.update({f"{k}_{tg:.2f}": v for k, v in s.items()})
    for phi in phis:
        for sdg in ("ego", "uniform"):
            s = DY.threshold_summary(csr, phi, sdg, runs["cc"], np.random.default_rng(seed_of("cc", ghash, phi, sdg)))
            out.update({f"cc_{sdg}_{k}_{phi:.2f}": v for k, v in s.items()})
    out.update(graph_hash=ghash)
    return out


def cmd_outcomes(net, seed):
    runs = dict(sir=RUNS54["sir"], cc=RUNS54["cc"])
    t, n, e0 = real_graph(net); dd = OUT_DIR / net; dd.mkdir(parents=True, exist_ok=True)
    cal = json.loads((dd / "calib.json").read_text())
    rows = []
    for arm in ARMS:
        for c in CELLS:
            r = run_record(net, arm, c, seed)
            if r is None or not len(r["edges"]):
                continue
            G = nx_of(n, r["edges"].tolist()); gh = oracle().graph_hash(G)
            o = outcomes54(G, t, cal, runs, gh)
            rows.append(dict(net=net, seed=seed, arm=ARM_NAME[arm], dq=DQ[c], graph=r["graph"], reached=r["reached"], stop=r["stop"],
                             depth=r["depth"], oracle=r["oracle"], t_wall_edit=r["t_wall"], start_hash=r["start_hash"],
                             **drift({k: o[k] for k in o if k.startswith("m_")}, t), **{k: v for k, v in o.items() if not isinstance(v, dict)}))
    pd.DataFrame(rows).to_csv(dd / f".seed{seed}.{os.getpid()}.csv", index=False); os.replace(dd / f".seed{seed}.{os.getpid()}.csv", dd / f"seed{seed}.csv")
    print(f"[outcomes] {net} seed {seed}: {len(rows)} graphs", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "threshold", "outcomes"])
    ap.add_argument("--net", required=True, choices=NETS); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "calibrate":
        cmd_calibrate(a.net)
    elif a.cmd == "threshold":
        cmd_threshold(a.net)
    else:
        cmd_outcomes(a.net, a.seed)


if __name__ == "__main__":
    main()
