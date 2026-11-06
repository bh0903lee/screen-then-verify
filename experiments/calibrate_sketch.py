"""Sketch accuracy per candidate and the spectral calibration slope kappa (training-sampler seeds)."""
import argparse, os, random, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
import run_experiment as M, operators as P, sketch as SK, graph_env as base, io_utils
from warmstart import metrics as CM

KEYS = base.KEYS


def timed_oracle(G):
    t = {}; m = {}
    for k in KEYS:
        t0 = time.perf_counter(); m[k] = CM.METRIC_FUNCTIONS[k](G); t[k] = time.perf_counter() - t0
    return m, t


def run(arm, seed, n_targets, n_steps, tag, k_big=16, split="eval", source="mix7", warm="ws", kband=None, noise_k=None, families=None):
    cfg = M.Cfg(source=source, warm=warm, kband=tuple(kband) if kband else P.KBAND, noise_k=noise_k, **({"families": list(families)} if families else {}))
    cfg.arm_weights = M.arm_weights_of(arm); groups = P.arm_groups(arm); rows = []
    if split == "train" and warm == "noised" and source == "mix7":
        import targets as PT
        tt = PT.sample_targets(n_targets, 1000 + seed, families=list(families) if families else PT.FAMILIES, with_graph=True); et = []
    else:
        tt, et = M.targets_for(cfg, seed)
    et = tt if split == "train" else et
    for ti, (t, fam) in enumerate(et[:n_targets]):
        env = M.make_eval_env(t, cfg); sd = 7000 + seed + ti
        G, res, _ = M.reset_connected(env, sd); G = G.copy(); rng = random.Random(sd); rng_sk = random.Random(sd * 7 + 1)
        n = G.number_of_nodes(); acc = {}
        for step in range(n_steps):
            ops, _ = M.candidates(G, groups, res, rng, cfg, with_fied=True, acc=acc, env=env)
            if not ops:
                continue
            W0 = M.worst(res); bind = KEYS[int(np.argmax(np.abs(res)))]
            sk = SK.state_sketch(G, n, rng_sk, k_src=k_big)
            m0, _ = timed_oracle(G); tgt = env.targets
            u_vec = sk["y"] * np.sqrt(np.array([G.degree(v) for v in range(n)], float))
            cand_rows = []
            for ci, op in enumerate(ops):
                cd16 = SK.cand_deltas(G, op, sk, res); cd8 = SK.cand_deltas(G, op, sk, res, k_src=8)
                dcomb = SK.dlam2_combinatorial(op, u_vec)
                P.apply_op(G, op); m1, tor = timed_oracle(G); lam_exact1 = SK.fiedler_data(G, n)[1]; P.revert_op(G, op)
                r1 = np.array([(m1[k] - tgt[k]) / base.TOL[k] for k in KEYS], np.float64); W1 = float(np.abs(r1).max())
                row = dict(arm=arm, seed=seed, target=ti, family=fam, step=step, cand=ci, op=op[0], kind=(op[2] if len(op) > 2 else ""),
                           W0=W0, bind0=bind, gap=sk["gap"], lam2=sk["lam2"], W1=W1, What8=cd8["W_hat"], What16=cd16["W_hat"],
                           t_oracle=sum(tor.values()), t_sketch16=cd16["t_sketch"], t_sketch8=cd8["t_sketch"],
                           t_state=sk["t_total"], t_state_fied=sk["t_fied"], t_state_comm=sk["t_comm"], t_state_bfs=sk["t_bfs"])
                for k in KEYS:
                    row[f"d_{k}"] = m1[k] - m0[k]; row[f"dh_{k}"] = float(cd16["d"][KEYS.index(k)]); row[f"t_or_{k}"] = tor[k]
                    row[f"rh_{k}"] = float(cd8["res_hat"][KEYS.index(k)])
                row["dh8_avg_path_length"] = cd8["dell"]; row["dh_comb_spectral_gap"] = dcomb
                row["d_spectral_gap_exact"] = lam_exact1 - sk["lam2"]
                cand_rows.append(row)
            rows += cand_rows
            pool = ops if len(ops) <= cfg.greedy_k else rng.sample(ops, cfg.greedy_k)
            op, bsc = None, 1e18
            for cand in pool:
                w1 = M.worst(M.eval_op(env, G, cand))
                if w1 < bsc:
                    bsc, op = w1, cand
            P.apply_op(G, op); res = env.residuals(G)
        print(f"  target {ti} ({fam}) done: {len(rows)} rows", flush=True)
    df = pd.DataFrame(rows); out = io_utils.RES / f"{M.EXP}_{tag}_sketch_acc_{arm}_s{seed}.csv"
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp"); df.to_csv(tmp, index=False); os.replace(tmp, out)
    print("wrote", out, len(df)); return df


def summary(tag, out=None, exclude_family=None, out_tag=None, fig_dir=None):
    _summary(tag, exclude_family=exclude_family, out_tag=out_tag)


def _summary(tag, exclude_family=None, out_tag=None, fig_dir=None):
    from scipy.stats import spearmanr
    fs = sorted(io_utils.RES.glob(f"{M.EXP}_{tag}_sketch_acc_*_s*.csv"))
    if not fs:
        sys.exit("no per-(arm, seed) files")
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); otag = out_tag or tag
    if exclude_family:
        assert "family" in df and exclude_family in set(df.family), f"family {exclude_family!r} not in {sorted(set(df.get('family', [])))}"
        assert out_tag and out_tag != tag, "--exclude-family needs --out-tag different from --tag (never overwrite the full-sample files)"
        n0 = len(df); df = df[df.family != exclude_family].reset_index(drop=True)
        print(f"LOFO: dropped family {exclude_family}: {n0 - len(df)} of {n0} candidate rows; outputs under tag {otag}\n")
    if not exclude_family:
        df.to_csv(io_utils.RES / f"{M.EXP}_{otag}_sketch_acc.csv", index=False)
    out = []
    for keys, d in df.groupby(["arm", "seed", "target", "step"]):
        if len(d) < 3 or d.W1.std() < 1e-12:
            continue
        w = d.W1.values; best = w.min(); rnd = w.mean() - best
        rec = dict(arm=keys[0], seed=keys[1], target=keys[2], step=keys[3], n=len(d), bind0=d.bind0.iloc[0], gap=d.gap.iloc[0], W0=d.W0.iloc[0])
        for kname in ("What8", "What16"):
            h = d[kname].values; i = int(np.argmin(h))
            rec[f"rho_{kname}"] = spearmanr(h, w).correlation if np.std(h) > 0 else 0.0
            rec[f"top1_{kname}"] = int(abs(w[i] - best) < 1e-12); rec[f"regret_{kname}"] = w[i] - best
        idx = np.arange(len(w)); rs = np.random.default_rng(int(keys[3]) * 1000 + int(keys[2]))
        rec["regret_random"] = rnd; rec["regret_greedy8"] = float(np.mean([w[rs.choice(idx, min(8, len(w)), replace=False)].min() - best for _ in range(20)]))
        out.append(rec)
    xs = df.d_spectral_gap.values; ys = df.dh_spectral_gap.values; lam_slope = float(np.polyfit(ys, xs, 1)[0]) if ys.std() > 0 else 1.0
    df["What8_cal"] = np.nan
    if all(f"rh_{k}" in df for k in KEYS):
        rh = np.stack([df[f"rh_{k}"].values for k in KEYS], 1).copy()
        j = KEYS.index("spectral_gap"); rh[:, j] = rh[:, j] - df.dh_spectral_gap.values / base.TOL["spectral_gap"] + lam_slope * df.dh_spectral_gap.values / base.TOL["spectral_gap"]
        df["What8_cal"] = np.abs(rh).max(1)
    st = pd.DataFrame(out)
    if "What8_cal" in df and df.What8_cal.notna().any():
        cal = []
        for keys, d in df.groupby(["arm", "seed", "target", "step"]):
            if len(d) < 3 or d.W1.std() < 1e-12:
                continue
            w = d.W1.values; h = d.What8_cal.values; i = int(np.argmin(h))
            cal.append(dict(arm=keys[0], seed=keys[1], target=keys[2], step=keys[3], rho_What8_cal=spearmanr(h, w).correlation if np.std(h) > 0 else 0.0, regret_What8_cal=w[i] - w.min()))
        st = st.merge(pd.DataFrame(cal), on=["arm", "seed", "target", "step"], how="left")
    if not exclude_family:
        st.to_csv(io_utils.RES / f"{M.EXP}_{otag}_sketch_acc_steps.csv", index=False)
    print(f"λ₂ first-order calibration slope (in-sample): {lam_slope:.3f}\n")
    met = []
    for k in KEYS:
        d = df[[f"d_{k}", f"dh_{k}"]].dropna(); x = d[f"d_{k}"].values; y = d[f"dh_{k}"].values
        r2 = 1 - ((x - y) ** 2).sum() / max(((x - x.mean()) ** 2).sum(), 1e-18)
        nz = np.abs(x) >= base.TOL[k] / 300
        met.append(dict(metric=k, n=len(d), R2=r2, corr=np.corrcoef(x, y)[0, 1], slope=float(np.polyfit(y, x, 1)[0]) if y.std() > 0 else np.nan,
                        sign_agree=float(np.mean(np.sign(x[nz]) == np.sign(y[nz]))) if nz.any() else np.nan,
                        MAE_over_eps=np.mean(np.abs(x - y)) / base.TOL[k], sd_over_eps=x.std() / base.TOL[k]))
    for col, k, name, xcol in (("dh8_avg_path_length", "avg_path_length", "avg_path_length_k8", "d_avg_path_length"),
                               ("dh_comb_spectral_gap", "spectral_gap", "spectral_gap_comb", "d_spectral_gap"),
                               ("dh_spectral_gap", "spectral_gap", "spectral_gap_vs_exact_eigh", "d_spectral_gap_exact"),
                               ("d_spectral_gap", "spectral_gap", "oracle_eigsh_vs_exact_eigh", "d_spectral_gap_exact")):
        if xcol not in df or col not in df:
            continue
        x = df[xcol].values; y = df[col].values; r2 = 1 - ((x - y) ** 2).sum() / max(((x - x.mean()) ** 2).sum(), 1e-18)
        nz = np.abs(x) >= base.TOL[k] / 300
        met.append(dict(metric=name, n=len(df), R2=r2, corr=np.corrcoef(x, y)[0, 1], slope=float(np.polyfit(y, x, 1)[0]) if y.std() > 0 else np.nan,
                        sign_agree=float(np.mean(np.sign(x[nz]) == np.sign(y[nz]))) if nz.any() else np.nan,
                        MAE_over_eps=np.mean(np.abs(x - y)) / base.TOL[k], sd_over_eps=x.std() / base.TOL[k]))
    for arm_, da in df.groupby("arm"):
        x = da.d_spectral_gap.values; y = da.dh_spectral_gap.values
        if y.std() > 0 and len(da) >= 50:
            met.append(dict(metric=f"spectral_gap@{arm_}", n=len(da), R2=np.nan, corr=np.corrcoef(x, y)[0, 1], slope=float(np.polyfit(y, x, 1)[0]),
                            sign_agree=np.nan, MAE_over_eps=np.mean(np.abs(x - y)) / base.TOL["spectral_gap"], sd_over_eps=x.std() / base.TOL["spectral_gap"]))
    mt = pd.DataFrame(met); _mo = io_utils.RES / f"{M.EXP}_{otag}_sketch_acc_metrics.csv"
    _mt = _mo.with_name(f".{_mo.name}.{os.getpid()}.tmp"); mt.to_csv(_mt, index=False); os.replace(_mt, _mo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="S_tswap"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--targets", type=int, default=12); ap.add_argument("--steps", type=int, default=30); ap.add_argument("--tag", default="s1")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--exclude-family", default=None)
    ap.add_argument("--fig-dir", default=None)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--nodes", type=int, default=200)
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--source", default="mix7")
    ap.add_argument("--oracle-lam", default="exact", choices=["eigsh", "eigh", "exact"])
    ap.add_argument("--warm", default="ws"); ap.add_argument("--kband", type=float, nargs=2, default=None)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--oracle-apl", default="nx", choices=["nx", "igraph"])
    a = ap.parse_args()
    M.set_nodes(a.nodes)
    M.set_oracle_lam(a.oracle_lam)
    M.set_oracle_apl(a.oracle_apl)
    if a.summary:
        summary(a.tag, None, exclude_family=a.exclude_family, out_tag=a.out_tag, fig_dir=a.fig_dir)
    else:
        warm, nk = M.parse_warm(a.warm)
        run(a.arm, a.seed, a.targets, a.steps, a.tag, split=a.split, source=a.source, warm=warm, kband=a.kband, noise_k=nk, families=a.families)
