"""Fixed candidate pools labelled by the oracle (ranking analysis and training data)."""
import argparse, random, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
import run_experiment as M, operators as P, sketch as SK, graph_env as base, io_utils

KEYS = base.KEYS
BS = (1, 4, 8, 16)
MARK = "<!-- ANALYSIS -->"


def rank_metrics(score, w, bs=BS):
    n = len(w); order = np.argsort(score, kind="stable"); best = np.argsort(w, kind="stable"); out = {}
    for b in bs:
        if b > n:
            continue
        top = set(order[:b].tolist())
        out[f"recall{b}"] = len(top & set(best[:b].tolist())) / b
        out[f"hit{b}"] = float(int(best[0]) in top)
        out[f"regret{b}"] = float(w[order[:b]].min() - w.min())
    return out


def collect(arm, seed, n_targets, steps, kmax, tag, split="eval", lam_scale="auto:sketch_calibration__N200_seeds20-22", warm_cache=None, target_cache=None, save_pools=False):
    M.set_oracle_lam("exact")
    cfg = M.Cfg(K=kmax, oracle_lam="exact", warm_cache=warm_cache, target_cache=target_cache); cfg.arm_weights = M.arm_weights_of(arm); groups = P.arm_groups(arm)
    kappa = M.resolve_lam_scale(lam_scale, arm=arm, nodes=200)
    tt, et = M.targets_for(cfg, seed); et = tt if split == "train" else et
    cand_rows = []; feats = []; pools = {}; t0 = time.time(); last = max(steps)
    for ti, (t, fam) in enumerate(et[:n_targets]):
        env = M.make_eval_env(t, cfg); sd = 7000 + seed + ti
        G, res, _ = M.reset_connected(env, sd); G = G.copy(); rng = random.Random(sd); rng_sk = random.Random(sd * 7 + 1); n = G.number_of_nodes()
        cfg8 = M.Cfg(K=32, oracle_lam="exact"); cfg8.arm_weights = cfg.arm_weights
        for step in range(last + 1):
            if step in steps:
                ops, fied = M.candidates(G, groups, res, rng, cfg, with_fied=True, acc={}, env=env)
                if len(ops) >= 8:
                    perm = list(range(len(ops))); random.Random(sd * 31 + step).shuffle(perm)
                    sk = SK.state_sketch(G, n, rng_sk); S = P.degree_stats(G); W0 = M.worst(res); bind = KEYS[int(np.argmax(np.abs(res)))]
                    if save_pools:
                        pools[(ti, step)] = dict(n=n, edges=np.asarray(sorted(G.edges()), np.int32), ops=[ops[ci] for ci in perm])
                    for pos, ci in enumerate(perm):
                        op = ops[ci]; cd = SK.cand_deltas(G, op, sk, res, lam_scale=kappa)
                        f = np.asarray(P.ext_features(G, op, S, fied, W0)[0], np.float32)
                        r1 = np.asarray(M.eval_op(env, G, op), np.float64)
                        cand_rows.append(dict(arm=arm, seed=seed, target=ti, family=fam, step=step, pos=pos, op=op[0], W0=W0, bind0=bind, gap=sk["gap"],
                                              W1=float(np.abs(r1).max()), What=float(cd["W_hat"]), n_pool=len(ops),
                                              **{f"r0_{k}": float(res[i]) for i, k in enumerate(KEYS)}, **{f"r1_{k}": float(r1[i]) for i, k in enumerate(KEYS)},
                                              **{f"rh_{k}": float(cd["res_hat"][i]) for i, k in enumerate(KEYS)}))
                        feats.append(f)
            if step == last:
                break
            ops8 = M.candidates(G, groups, res, rng, cfg8, acc={}, env=env)
            if not ops8:
                break
            pool = ops8 if len(ops8) <= 8 else rng.sample(ops8, 8); op, bsc, rsel = None, 1e18, None
            for c in pool:
                r = M.eval_op(env, G, c); w = M.worst(r)
                if w < bsc:
                    bsc, op, rsel = w, c, r
            P.apply_op(G, op); res = np.asarray(rsel, np.float32)
        print(f"  target {ti} ({fam}) done: {len(cand_rows)} candidate rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(cand_rows); F = np.vstack(feats) if feats else np.zeros((0, 1), np.float32)
    assert len(df) == len(F)
    y = (df.W1 - df.W0).values; pred = np.full(len(df), np.nan)
    for ti in sorted(df.target.unique()):
        te = (df.target == ti).values
        if te.all() or (~te).sum() < 50:
            continue
        pred[te] = M.fit_ridge(F[~te], y[~te]).predict(F[te])
    df["ridge_dW"] = pred
    od = io_utils.RES / f"{M.EXP}_{tag}"; od.mkdir(parents=True, exist_ok=True)
    df.to_csv(od / f"cand_{arm}_s{seed}.csv", index=False); np.savez_compressed(od / f"feat_{arm}_s{seed}.npz", F=F)
    if save_pools:
        import pickle
        assert sum(len(v["ops"]) for v in pools.values()) == len(df), "pool ops must match candidate rows one-to-one"
        (od / f"pools_{arm}_s{seed}.pkl").write_bytes(pickle.dumps(pools, protocol=4))
    rows = []
    for (ti, step), d in df.groupby(["target", "step"]):
        d = d.sort_values("pos"); rr = np.random.default_rng(7000 + seed * 131 + ti * 17 + step)
        for K in sorted({k for k in (32, 64, kmax) if k <= len(d)} | ({len(d)} if len(d) < 32 else set())):
            x = d.iloc[:K]; w = x.W1.values
            if w.std() < 1e-12:
                continue
            rules = {"oracle": w, "sketch": x.What.values, "random": rr.random(K)}
            if x.ridge_dW.notna().all():
                rules["ridge"] = x.ridge_dW.values
            srt = np.sort(w)
            for rule, sc in rules.items():
                rec = dict(arm=arm, seed=seed, target=ti, family=d.family.iloc[0], step=step, K=K, rule=rule, bind0=d.bind0.iloc[0], W0=float(d.W0.iloc[0]), gap=float(d.gap.iloc[0]),
                           best_gain=float(d.W0.iloc[0] - w.min()), margin=float(srt[1] - srt[0]), **rank_metrics(sc, w))
                if rule == "sketch":
                    from scipy.stats import spearmanr
                    err = np.abs(x.What.values - w); rec.update(spearman=float(spearmanr(sc, w).correlation) if np.std(sc) > 0 else 0.0,
                                                               eta_max=float(err.max()), eta95=float(np.quantile(err, 0.95)), eta_med=float(np.median(err)))
                    rec["bound_ok"] = float(rec["regret1"] <= 2 * rec["eta_max"] + 1e-9)
                rows.append(rec)
    out = pd.DataFrame(rows); out.to_csv(od / f"recall_{arm}_s{seed}.csv", index=False)
    (od / f"{arm}_seed{seed}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("wrote", od, len(df), "candidates", len(out), "recall rows"); return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm"); ap.add_argument("--seed", type=int); ap.add_argument("--targets", type=int, default=28)
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 6, 12, 18, 24, 30]); ap.add_argument("--kmax", type=int, default=128)
    ap.add_argument("--tag", default="candidate_pool__evaluation_states"); ap.add_argument("--split", default="eval", choices=("eval", "train"))
    ap.add_argument("--sketch-lam-scale", default="auto:sketch_calibration__N200_seeds20-22"); ap.add_argument("--warm-cache", default=None); ap.add_argument("--target-cache", default=None)
    ap.add_argument("--save-pools", action="store_true")
    a = ap.parse_args()
    collect(a.arm, a.seed, a.targets, sorted(a.steps), a.kmax, a.tag, split=a.split, lam_scale=a.sketch_lam_scale, warm_cache=a.warm_cache, target_cache=a.target_cache, save_pools=a.save_pools)
