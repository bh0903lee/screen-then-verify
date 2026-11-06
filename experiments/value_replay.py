"""Replay of trained value networks on fixed states."""
import argparse, random, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import run_experiment as M, operators as P, sketch as SK, graph_env as base, io_utils


def replay(tag, arm, seed, n_targets, n_steps, cfg, kind="value"):
    groups = P.arm_groups(arm); tt, et = M.targets_for(cfg, seed)
    ck = io_utils.load_checkpoint(f"{M.EXP}_{tag}", kind, seed, arm)
    if ck is None:
        raise FileNotFoundError(f"no {kind} checkpoint for {tag}/{arm}/seed{seed}")
    net = M.make_net(kind, cfg); net.load_state_dict(ck["state_dict"])
    rows = []
    for ti, (t, fam) in enumerate(et[:n_targets]):
        env = base.SwapEnv(M.spec_of(t), forward_sweep=True); sd = 7000 + seed + ti
        G, res, _ = M.reset_connected(env, sd); G = G.copy(); rng = random.Random(sd); n = G.number_of_nodes()
        rng_sk = random.Random(sd * 7 + 1)
        for step in range(n_steps):
            ops = M.candidates(G, groups, res, rng, cfg, env=env)
            if not ops:
                continue
            pool = ops if len(ops) <= cfg.value_k else rng.sample(ops, cfg.value_k)
            sk = SK.state_sketch(G, n, rng_sk, k_src=cfg.sketch_k) if kind == "value_sketch" else None
            S = P.degree_stats(G) if kind in ("value_af", "value_sketch") else None
            V, W, kinds, Vraw = [], [], [], []
            for c in pool:
                af = M.sketch_af(G, c, sk, res, S, cfg.sketch_lam_scale) if kind == "value_sketch" else (P.op_features(G, c, S)[0] if kind == "value_af" else None)
                P.apply_op(G, c); r1 = env.residuals(G); V.append(M.cand_score(net, G, n, r1, cfg, kind, af, fam)); W.append(M.worst(r1))
                Vraw.append(float(M.value_of(net, G, n, r1, cfg.d0, M.extra_vec(kind, cfg, af, fam))))
                P.revert_op(G, c); kinds.append(c[2].split(":")[-1] if ":" in c[2] else c[2])
            i_v = int(np.argmin(V)); i_w = int(np.argmin(W)); i_r = rng.randrange(len(pool))
            allk = [c[2].split(":")[-1] if ":" in c[2] else c[2] for c in ops]
            rows.append(dict(arm=arm, seed=seed, target=ti, family=fam, step=step, w0=M.worst(res),
                             n_pool=len(ops), n_sub=len(pool),
                             v_std=float(np.std(V)), w_std=float(np.std(W)),
                             spearman_VW=float(spearmanr(V, W).correlation) if len(pool) > 2 and np.std(V) > 0 else np.nan,
                             vraw_mean=float(np.mean(Vraw)), vraw_std=float(np.std(Vraw)), vraw_cv=float(np.std(Vraw) / max(abs(np.mean(Vraw)), 1e-9)),
                             spearman_VrawW=float(spearmanr(Vraw, W).correlation) if len(pool) > 2 and np.std(Vraw) > 0 else np.nan,
                             kind_value=kinds[i_v], kind_best=kinds[i_w], kind_random=kinds[i_r],
                             regret_value=W[i_v] - W[i_w], regret_random=W[i_r] - W[i_w],
                             dW_value=W[i_v] - M.worst(res), dW_random=W[i_r] - M.worst(res), dW_best=W[i_w] - M.worst(res),
                             sub_overlap=M.pool_diversity(pool), pool_overlap=M.pool_diversity(ops),
                             **{f"pool_frac_{k}": allk.count(k) / len(allk) for k in ("r", "tri", "comm", "spec", "free")},
                             **{f"sub_frac_{k}": kinds.count(k) / len(kinds) for k in ("r", "tri", "comm", "spec", "free")}))
            op = pool[i_v]
            P.apply_op(G, op); res = env.residuals(G)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="b1"); ap.add_argument("--arm", default="S_tswap")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--targets", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--source", default="mix7")
    ap.add_argument("--value-label", default="clip3"); ap.add_argument("--value-fam", action="store_true")
    ap.add_argument("--kind", default="value", choices=["value", "value_rel", "value_af", "value_rel_res", "value_sketch"])
    ap.add_argument("--sketch-lam-scale", default="1.0")
    ap.add_argument("--sketch-k", type=int, default=8)
    ap.add_argument("--oracle-lam", default="exact", choices=["eigsh", "eigh", "exact"])
    a = ap.parse_args()
    M.set_oracle_lam(a.oracle_lam)
    cfg = M.Cfg(source=a.source, value_label=a.value_label, value_fam=a.value_fam, sketch_k=a.sketch_k, oracle_lam=a.oracle_lam,
                sketch_lam_scale=M.resolve_lam_scale(a.sketch_lam_scale, arm=a.arm))
    cfg.arm_weights = M.arm_weights_of(a.arm)
    df = pd.concat([replay(a.tag, a.arm, s, a.targets, a.steps, cfg, a.kind) for s in a.seeds], ignore_index=True)
    out = io_utils.RES / (f"{M.EXP}_{a.tag}_e6_{a.arm}.csv" if a.kind == "value" else f"{M.EXP}_{a.tag}_e6_{a.arm}_{a.kind}.csv"); df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} steps)")
    print("V std / W' std:", df.v_std.mean().round(4), "/", df.w_std.mean().round(4), "| Spearman(V,W') mean:", df.spearman_VW.mean().round(3))
    print("raw V: mean", df.vraw_mean.mean().round(4), "std", df.vraw_std.mean().round(4), "CV", df.vraw_cv.mean().round(4), "| Spearman(V_raw,W') mean:", df.spearman_VrawW.mean().round(3))
    print("per-step regret vs pool-best: value", df.regret_value.mean().round(4), " random", df.regret_random.mean().round(4))
    print("per-step ΔW: value", df.dW_value.mean().round(4), " random", df.dW_random.mean().round(4), " best", df.dW_best.mean().round(4))
    print("kind chosen by value:", df.kind_value.value_counts(normalize=True).round(3).to_dict())
    print("kind = pool best   :", df.kind_best.value_counts(normalize=True).round(3).to_dict())
    print("overlap: subset", df.sub_overlap.mean().round(4), " pool", df.pool_overlap.mean().round(4))


if __name__ == "__main__":
    main()
