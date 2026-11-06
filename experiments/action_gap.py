"""How often depth-3 lookahead changes the preferred candidate."""
import argparse
import os
import random
import sys
import time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
import run_experiment as M, operators as P, sketch as SK, io_utils

HS = (0, 3, 10, 30, 59)
GATE_REORDER, GATE_RATIO, GATE_HMIN = 0.10, 1.0, 10


def continuation(env, G0, res0, groups, cfg, rng, rng_sk, hmax, fam=None):
    G = G0.copy(); res = np.asarray(res0).copy(); n = G.number_of_nodes(); ws = []
    for _ in range(hmax):
        ops = M.candidates(G, groups, res, rng, cfg, env=env, fam=fam)
        if not ops:
            ws.append(ws[-1] if ws else M.worst(res)); continue
        sk = SK.state_sketch(G, n, rng_sk, k_src=cfg.sketch_k)
        what = [SK.cand_deltas(G, c, sk, res, lam_scale=cfg.sketch_lam_scale)["W_hat"] for c in ops]
        P.apply_op(G, ops[int(np.argmin(what))]); res = env.residuals(G); ws.append(M.worst(res))
    return ws


def run(arm, seed, cfg, n_targets=12, states=8, stride=3, reps=3, hmax=59, log=print):
    groups = P.arm_groups(arm); _, et = M.targets_for(cfg, seed); rows = []
    hs = [h for h in HS if h <= hmax] + ([hmax] if hmax not in HS else [])
    for ti, (t, fam) in enumerate(et[:n_targets]):
        t0 = time.time(); env = M.make_eval_env(t, cfg); sd = 7000 + seed + ti
        G, res, _ = M.reset_connected(env, sd); G = G.copy(); rng = random.Random(sd)
        for step in range(states * stride):
            ops = M.candidates(G, groups, res, rng, cfg, env=env, fam=fam)
            if not ops:
                continue
            pool = ops if len(ops) <= cfg.greedy_k else rng.sample(ops, cfg.greedy_k)
            R1 = [M.eval_op(env, G, c) for c in pool]; Wp = [M.worst(r) for r in R1]
            if step % stride == 0:
                sidx = step // stride
                for j, c in enumerate(pool):
                    Gj = G.copy(); P.apply_op(Gj, c)
                    for rep in range(reps):
                        crng = random.Random(sd * 1009 + sidx * 97 + j * 13 + rep)
                        csk = random.Random(sd * 2003 + sidx * 89 + j * 7 + rep)
                        ws = continuation(env, Gj, R1[j], groups, cfg, crng, csk, hmax, fam=fam)
                        runmin = np.minimum.accumulate([Wp[j]] + ws)
                        rows.append(dict(arm=arm, seed=seed, target=ti, family=fam, state=sidx, step=step, w_state=M.worst(res),
                                         cand=j, rep=rep, W1=Wp[j], **{f"V{h}": float(runmin[min(h, hmax)]) for h in hs}))
            i = int(np.argmin(Wp)); P.apply_op(G, pool[i]); res = env.residuals(G)
        log(f"[action-gap] {arm} seed{seed} target{ti} ({fam}): {sum(1 for r in rows if r['target'] == ti)} rows ({time.time()-t0:.0f}s)")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="S_tswap"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="s4")
    ap.add_argument("--targets", type=int, default=12); ap.add_argument("--states", type=int, default=8); ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--reps", type=int, default=3); ap.add_argument("--hmax", type=int, default=59)
    ap.add_argument("--oracle-lam", default="exact", choices=["eigsh", "eigh", "exact"])
    ap.add_argument("--sketch-lam-scale", default="auto")
    a = ap.parse_args()
    M.set_oracle_lam(a.oracle_lam)
    cfg = M.Cfg(sketch_lam_scale=M.resolve_lam_scale(a.sketch_lam_scale, arm=a.arm), no_signal=True, oracle_lam=a.oracle_lam); cfg.arm_weights = M.arm_weights_of(a.arm)
    t0 = time.time(); df = run(a.arm, a.seed, cfg, a.targets, a.states, a.stride, a.reps, a.hmax)
    out = io_utils.RES / f"run_{a.tag}_actiongap_{a.arm}_s{a.seed}.csv"; tmp = out.with_suffix(f".tmp{os.getpid()}")
    df.to_csv(tmp, index=False); os.replace(tmp, out)
    print(f"wrote {out} ({len(df)} rows, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
