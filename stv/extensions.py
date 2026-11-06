"""Selectors for the learner-scaling, required-ranking-quality, adaptive-verification and generic-surrogate experiments."""
import math, re
import numpy as np

import operators as P
import sketch as SK

PROBE_EVERY = 5
_RE = {"rho": re.compile(r"rho(0\.\d+|1\.0)(k8)?"), "rhov": re.compile(r"rhov(\d+(?:\.\d+)?)(k8)?"), "cv": re.compile(r"cv(0\.\d+)"),
       "skq": re.compile(r"skq(\d+)")}
FIXED = ("olr_k8", "ogb_k8", "olrm_k8", "mf_k8", "bsk_k8", "lsk_k8", "gbp", "gbp_k8", "gbps", "gbps_k8", "nnpred", "nnpred_k8", "nnpreds", "nnpreds_k8")
KEYS = SK.KEYS
I_LAM = KEYS.index("spectral_gap")


def kind(policy):
    p = str(policy)
    if p in FIXED:
        return p
    for k, rx in _RE.items():
        if rx.fullmatch(p):
            return k
    return None


def is_ext(policy):
    return kind(policy) is not None


def needs_sketch(policy):
    return kind(policy) in ("cv", "skq", "bsk_k8", "lsk_k8", "mf_k8", "gbps", "gbps_k8", "nnpreds", "nnpreds_k8")


def sigma_for_spearman(r):
    r = float(r)
    if r >= 1.0:
        return 0.0
    if not 0.0 < r < 1.0:
        raise ValueError(f"target Spearman must be in (0, 1], got {r}")
    rp = 2.0 * math.sin(math.pi * r / 6.0)
    return math.sqrt(1.0 / rp ** 2 - 1.0)


def _normal_scores(w, rng):
    from scipy.stats import norm
    k = len(w); order = np.lexsort((rng.random(k), np.asarray(w, np.float64)))
    rank = np.empty(k); rank[order] = np.arange(1, k + 1)
    return norm.ppf((rank - 0.5) / k)


def spearman(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.all(np.asarray(a) == a[0]) or np.all(np.asarray(b) == b[0]):
        return np.nan
    return float(spearmanr(a, b).statistic)


def _worst(r):
    return float(np.abs(np.asarray(r, np.float64)).max())


def probe(env, G, op):
    f = getattr(env, "_res_raw", None)
    if f is None:
        raise RuntimeError("probe needs env._res_raw (count_env must wrap the env first)")
    P.apply_op(G, op)
    try:
        return f(G)
    finally:
        P.revert_op(G, op)


def _verify(env, G, ops, idx, worst):
    best = None; bsc = 1e18; r_sel = None; seen = {}
    for i in idx:
        P.apply_op(G, ops[i]); r1 = env.residuals(G); P.revert_op(G, ops[i])
        sc = worst(r1); seen[int(i)] = (sc, np.asarray(r1, np.float64))
        if sc < bsc:
            bsc, best, r_sel = sc, ops[i], r1
    return best, r_sel, len(idx), seen


def _sketch_all(G, ops, sk, res, cfg):
    return [SK.cand_deltas(G, op, sk, res, lam_scale=cfg.sketch_lam_scale) for op in ops]


def _probe_pool(env, G, ops, worst, st=None):
    import time
    t0 = time.perf_counter(); Gp = G.copy(); w = np.array([worst(probe(env, Gp, op)) for op in ops], np.float64)
    if st is not None:
        st["t_probe"] += time.perf_counter() - t0
    return w


def _record(st, score, w):
    o = np.argsort(score, kind="stable"); best = float(w.min())
    st["rho"].append((st["step"], spearman(score, w)))
    st["hit1"].append(float(w[o[0]] <= best + 1e-12)); st["top8"].append(float(w[o[:8]].min() <= best + 1e-12))


def init_state(policy, sd, cfg, model=None):
    st = dict(policy=str(policy), kind=kind(policy), rng=np.random.default_rng(1_000_003 * (int(sd) + 1) + 17), step=0,
              n_ver=[], rho=[], hit1=[], top8=[], t_probe=0.0, cov_n=0, cov_hit=0, X=[], y=[], Y=[], fit=None, fit_n=0, model=model)
    if st["kind"] == "rho":
        m = _RE["rho"].fullmatch(str(policy)); st["r"] = float(m.group(1)); st["k8"] = bool(m.group(2))
        st["sigma"] = sigma_for_spearman(st["r"])
    elif st["kind"] == "rhov":
        m = _RE["rhov"].fullmatch(str(policy)); st["sigma"] = float(m.group(1)); st["k8"] = bool(m.group(2))
    elif st["kind"] == "cv":
        st["alpha"] = float(_RE["cv"].fullmatch(str(policy)).group(1))
        st["tau"] = cv_tau(cfg, st["alpha"])
    elif st["kind"] == "skq":
        st["q"] = int(_RE["skq"].fullmatch(str(policy)).group(1))
        if st["q"] < 1:
            raise ValueError(policy)
    return st


def cv_tau(cfg, alpha):
    import pandas as pd
    path = getattr(cfg, "ext_calib", None); arm = getattr(cfg, "ext_arm", None)
    if not path:
        raise ValueError("cv<α> policies need --ext-calib (output of experiments/conformal_thresholds.py)")
    t = pd.read_csv(path)
    row = t[(t.arm.astype(str) == str(arm)) & np.isclose(t.alpha.astype(float), alpha)]
    if len(row) != 1:
        raise ValueError(f"{path}: no unique τ for arm={arm} alpha={alpha}")
    tau = float(row.tau.iloc[0])
    if np.isnan(tau) or tau < 0:
        raise ValueError(f"{path}: implausible τ={tau} for arm={arm} alpha={alpha}")
    return tau


def _local_X(G, ops, fied, res):
    import predictors as PR
    S_ = P.degree_stats(G); w0 = _worst(res)
    Fl = [P.ext_features(G, c, S_, fied, w0)[0] for c in ops]
    return PR.build_X(Fl, res, w0), Fl, w0


def _ridge_fit(X, y, lam=1.0):
    mu = X.mean(0); sd = X.std(0) + 1e-8; Z = (X - mu) / sd
    A = Z.T @ Z + lam * np.eye(Z.shape[1]); b = Z.T @ (y - y.mean())
    w = np.linalg.solve(A, b)
    return ("ridge", mu, sd, w, float(y.mean()))


def _predict(fit, X):
    if fit[0] == "ridge":
        _, mu, sd, w, b0 = fit
        return ((X - mu) / sd) @ w + b0
    return fit[1].predict(X)


def choose(policy, st, G, ops, fied, res, env, rng, cfg, sk, worst):
    k = st["kind"]; st["step"] += 1; probe_now = (st["step"] - 1) % PROBE_EVERY == 0
    vk = int(cfg.value_k)
    if k in ("rho", "rhov"):
        w = _probe_pool(env, G, ops, worst, st)
        if k == "rho":
            base_ = _normal_scores(w, st["rng"])
        else:
            base_ = (w - w.mean()) / (w.std() + 1e-12)
        s = base_ + st["sigma"] * st["rng"].standard_normal(len(w))
        _record(st, s, w); order = np.argsort(s, kind="stable")
        if not st["k8"]:
            i = int(order[0]); st["n_ver"].append(0)
            return ops[i], None, 0
        op, r_sel, n, _ = _verify(env, G, ops, order[:vk], worst); st["n_ver"].append(n)
        return op, r_sel, n
    if k in ("cv", "skq", "bsk_k8", "lsk_k8"):
        cds = _sketch_all(G, ops, sk, res, cfg); st["n_sk"] = st.get("n_sk", 0) + len(ops)
        if k in ("cv", "skq"):
            score = np.array([c["W_hat"] for c in cds], np.float64)
        elif k == "bsk_k8":
            b = int(np.argmax(np.abs(np.asarray(res, np.float64))))
            score = np.array([abs(float(c["res_hat"][b])) for c in cds], np.float64)
        else:
            score = np.array([abs(float(c["res_hat"][I_LAM])) for c in cds], np.float64)
        order = np.argsort(score, kind="stable")
        if k == "cv":
            sel = [int(i) for i in order if score[i] <= score[order[0]] + st["tau"]][:len(ops)] or [int(order[0])]
        elif k == "skq":
            sel = [int(i) for i in order[:st["q"]]]
        else:
            sel = [int(i) for i in order[:vk]]
        op, r_sel, n, _ = _verify(env, G, ops, sel, worst); st["n_ver"].append(n)
        if probe_now:
            w = _probe_pool(env, G, ops, worst, st); _record(st, score, w)
            if k == "cv":
                st["cov_n"] += 1; st["cov_hit"] += int(int(np.argmin(w)) in set(sel) or float(w.min()) >= min(w[i] for i in sel) - 1e-12)
        return op, r_sel, n
    if k in ("olr_k8", "ogb_k8", "olrm_k8", "mf_k8"):
        X, Fl, w0 = _local_X(G, ops, fied, res)
        if k == "mf_k8":
            import predictors as PR
            cds = _sketch_all(G, ops, sk, res, cfg); st["n_sk"] = st.get("n_sk", 0) + len(ops)
            what = np.array([c["W_hat"] for c in cds], np.float64)
            X = PR.build_X(Fl, res, w0, np.array([c["res_hat"] for c in cds], np.float32), what.astype(np.float32))
        need = 50 if k == "ogb_k8" else 20
        if len(st["y"]) >= need:
            if k in ("olr_k8", "mf_k8"):
                st["fit"] = _ridge_fit(np.asarray(st["X"]), np.asarray(st["y"]))
            elif k == "olrm_k8":
                st["fit"] = [_ridge_fit(np.asarray(st["X"]), np.asarray(st["Y"])[:, m]) for m in range(len(KEYS))]
            elif st["fit"] is None or st["step"] - st["fit_n"] >= 5:
                from sklearn.ensemble import HistGradientBoostingRegressor as HGB
                m = HGB(max_iter=100, random_state=0).fit(np.asarray(st["X"]), np.asarray(st["y"]))
                st["fit"] = ("hgb", m); st["fit_n"] = st["step"]
        if st["fit"] is not None:
            score = (np.max(np.abs(np.stack([_predict(f, X) for f in st["fit"]], 1)), 1) if k == "olrm_k8" else _predict(st["fit"], X))
        elif k == "mf_k8":
            score = what
        else:
            score = None
        if score is not None:
            order = np.argsort(score, kind="stable"); sel = [int(i) for i in order[:vk]]
        else:
            sel = sorted(int(i) for i in st["rng"].choice(len(ops), size=min(vk, len(ops)), replace=False))
        op, r_sel, n, seen = _verify(env, G, ops, sel, worst); st["n_ver"].append(n)
        for i, (wv, rv) in seen.items():
            st["X"].append(X[i]); st["y"].append(wv - w0); st["Y"].append(rv)
        if probe_now and score is not None:
            w = _probe_pool(env, G, ops, worst, st); _record(st, score, w)
        return op, r_sel, n
    if k in ("nnpred", "nnpred_k8", "nnpreds", "nnpreds_k8", "gbp", "gbp_k8", "gbps", "gbps_k8"):
        import predictors as PR
        X, Fl, w0 = _local_X(G, ops, fied, res); rh = what = None
        if k in ("nnpreds", "nnpreds_k8", "gbps", "gbps_k8"):
            cds = _sketch_all(G, ops, sk, res, cfg); st["n_sk"] = st.get("n_sk", 0) + len(ops)
            rh = np.array([c["res_hat"] for c in cds], np.float32); what = np.array([c["W_hat"] for c in cds], np.float32)
        if k.startswith("gbp"):
            score = np.asarray(st["model"].predict(PR.build_X(Fl, res, w0, rh, what)), np.float64)
        else:
            score = np.asarray(st["model"].score(G, ops, Fl, res, rh, what), np.float64)
        order = np.argsort(score, kind="stable")
        if probe_now:
            w = _probe_pool(env, G, ops, worst, st); _record(st, score, w)
        if not k.endswith("k8"):
            st["n_ver"].append(0); return ops[int(order[0])], None, 0
        op, r_sel, n, _ = _verify(env, G, ops, [int(i) for i in order[:vk]], worst); st["n_ver"].append(n)
        return op, r_sel, n
    raise ValueError(f"not an extension policy: {policy}")


def summary(st):
    rho = [x for _, x in st["rho"] if x == x]; late = [x for t, x in st["rho"] if x == x and t > 30]
    m = lambda v: round(float(np.mean(v)), 4) if v else np.nan
    out = dict(ext_q_mean=round(float(np.mean(st["n_ver"])), 4) if st["n_ver"] else 0.0,
               ext_rho_real=m(rho), ext_rho_late=m(late), ext_rho_n=len(rho), ext_hit1=m(st["hit1"]), ext_top8=m(st["top8"]),
               ext_t_probe=round(st["t_probe"], 3), ext_sketch_cand=int(st.get("n_sk", 0)))
    if st["kind"] == "rhov":
        out["ext_rhov_sigma"] = st["sigma"]
    if st["kind"] == "rho":
        out["ext_rho_target"] = st["r"]
    if st["kind"] == "cv":
        out.update(ext_tau=round(st["tau"], 6), ext_alpha=st["alpha"], ext_cov_n=st["cov_n"],
                   ext_cov=round(st["cov_hit"] / st["cov_n"], 4) if st["cov_n"] else np.nan)
    return out


def load_model(policy, cfg, arm):
    k = kind(policy); obj = "top8" if str(policy).endswith("k8") else "top1"
    if k in ("gbp", "gbp_k8", "gbps", "gbps_k8"):
        import predictors as PR
        return PR.load(cfg.pred_dir, arm, "LS" if k.startswith("gbps") else "L", obj)
    if k not in ("nnpred", "nnpred_k8", "nnpreds", "nnpreds_k8"):
        return None
    import edit_head
    path = getattr(cfg, "ext_model", None)
    if not path:
        raise ValueError("nnpred policies need --ext-model")
    m = edit_head.load(str(path).format(arm=arm, obj=obj))
    if (m.meta.get("feat", "L") == "LS") != k.startswith("nnpreds"):
        raise ValueError(f"{policy}: model {path} was trained with features {m.meta.get('feat', 'L')}")
    return m


def model_digest(path_tmpl, arm, policy):
    import hashlib
    from pathlib import Path
    p = Path(str(path_tmpl).format(arm=arm, obj=("top8" if str(policy).endswith("k8") else "top1")))
    return hashlib.md5(p.read_bytes()).hexdigest()[:12] if p.exists() else "missing"
