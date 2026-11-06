"""Train the boosting direct predictors on candidate pools."""
import argparse, pickle, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
import io_utils, predictors as PR
from candidate_pool import rank_metrics

EXP = "run"
VAL_SEEDS = (26, 27)
DEPLOY_K = 32


def load(arm, tag, exclude_family=None, seeds=None):
    od = io_utils.RES / f"{EXP}_{tag}"; dfs, Fs = [], []
    for f in sorted(od.glob(f"cand_{arm}_s*.csv")):
        sd = int(f.stem.rsplit("_s", 1)[1])
        if seeds is not None and sd not in seeds:
            continue
        d = pd.read_csv(f); F = np.load(od / f"feat_{arm}_s{sd}.npz")["F"]
        assert len(d) == len(F), f"{f}: {len(d)} rows vs {len(F)} feature rows"
        dfs.append(d); Fs.append(F)
    if not dfs:
        sys.exit(f"no training data for {arm} under {od}")
    df, F = pd.concat(dfs, ignore_index=True), np.vstack(Fs)
    if exclude_family:
        assert exclude_family in set(df.family), f"family {exclude_family!r} not in the table ({sorted(set(df.family))})"
        keep = (df.family != exclude_family).values; df, F = df[keep].reset_index(drop=True), F[keep]
        print(f"  LOFO: excluded family {exclude_family}: {int((~keep).sum())} rows dropped, {len(df)} kept", flush=True)
    return df, F


def candidates_grid():
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    g = [(f"ridge_a{a}", lambda a=a: make_pipeline(StandardScaler(), Ridge(alpha=a))) for a in (0.1, 10.0)]
    g += [(f"hgb_i{it}_lr{lr}_d{dp}", lambda it=it, lr=lr, dp=dp: HGB(max_iter=it, learning_rate=lr, max_depth=dp, random_state=0))
          for it in (200, 500) for lr in (0.05, 0.1) for dp in (None, 6)]
    g += [(f"mlp_{'x'.join(map(str, h))}", lambda h=h: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=h, early_stopping=True, max_iter=300, random_state=0)))
          for h in ((128, 64), (256, 128, 64))]
    return g


def state_scores(df, pred):
    from scipy.stats import spearmanr
    rows = []
    for _, idx in df.groupby(["seed", "target", "step"]).indices.items():
        w = df.W1.values[idx]
        if len(idx) < 16 or w.std() < 1e-12:
            continue
        r = rank_metrics(pred[idx], w); r["spearman"] = float(spearmanr(pred[idx], w).correlation) if np.std(pred[idx]) > 0 else 0.0
        rows.append(r)
    return pd.DataFrame(rows).mean().to_dict() if rows else {}


def train(arm, tag, out, exclude_family=None, feats=None, seeds=None, refit_val=True):
    df, F = load(arm, tag, exclude_family, seeds); out = Path(out); out.mkdir(parents=True, exist_ok=True); feats = list(feats) if feats else list(PR.FEATS)
    y = (df.W1 - df.W0).values.astype(np.float64); tr = ~df.seed.isin(VAL_SEEDS).values
    va = df.seed.isin(VAL_SEEDS).values & (df.pos.values < DEPLOY_K)
    assert tr.sum() > 1000 and va.sum() > 200, f"too little data: train {tr.sum()}, val {va.sum()}"
    report = []
    for feat in feats:
        X = PR.build_X_table(F, df, feat); best = {"top1": None, "top8": None}
        for name, make in candidates_grid():
            t0 = time.time(); m = make().fit(X[tr], y[tr]); p = m.predict(X[va]); sc = state_scores(df[va].reset_index(drop=True), np.asarray(p))
            rec = dict(arm=arm, feat=feat, model=name, n_train=int(tr.sum()), n_val=int(va.sum()), fit_s=round(time.time() - t0, 1),
                       val_rmse=float(np.sqrt(np.mean((p - y[va]) ** 2))), **{f"val_{k}": round(float(v), 4) for k, v in sc.items()})
            report.append(rec)
            keys = {"top1": (rec.get("val_regret1", 9e9), -rec.get("val_hit1", 0.0)), "top8": (rec.get("val_regret8", 9e9), -rec.get("val_recall8", 0.0))}
            for obj, key in keys.items():
                if best[obj] is None or key < best[obj][0]:
                    best[obj] = (key, name, make)
            print(f"  {arm} {feat} {name}: regret8 {rec.get('val_regret8')}, recall8 {rec.get('val_recall8')}, spearman {rec.get('val_spearman')} ({rec['fit_s']}s)", flush=True)
        dv = df[va].reset_index(drop=True)
        for ref, sc_ in (("ref_sketch", dv.What.values), ("ref_random", np.random.default_rng(0).random(len(dv)))):
            report.append(dict(arm=arm, feat=feat, model=ref, n_train=0, n_val=int(va.sum()), fit_s=0.0, val_rmse=np.nan, **{f"val_{k}": round(float(v), 4) for k, v in state_scores(dv, sc_).items()}))
        for obj in ("top1", "top8"):
            fit_rows = np.ones(len(y), bool) if refit_val else tr
            t0 = time.time(); final = best[obj][2]().fit(X[fit_rows], y[fit_rows])
            meta = dict(selected=best[obj][1], objective=obj, deploy_k=DEPLOY_K, n_fit=int(fit_rows.sum()), refit_val=bool(refit_val), n_features=int(X.shape[1]), fit_s=round(time.time() - t0, 1),
                        label_oracle_calls=int(len(y)), val_seeds=list(VAL_SEEDS), tag=tag, exclude_family=exclude_family, trained=time.strftime("%Y-%m-%d %H:%M:%S"))
            PR.model_path(out, arm, feat, obj).write_bytes(pickle.dumps(dict(model=final, feat=feat, arm=arm, meta=meta)))
            print(f"  → {arm} {feat} [{obj}]: selected {best[obj][1]} (n_fit {len(y)}, {X.shape[1]} features)", flush=True)
    pd.DataFrame(report).to_csv(out / f"val_{arm}.csv", index=False)
    (out / f"train_{arm}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--tag", default="training_pools__direct_predictors"); ap.add_argument("--out", default=str(io_utils.RES / "models" / "boosting"))
    ap.add_argument("--exclude-family", default=None)
    ap.add_argument("--feats", nargs="+", default=None, choices=list(PR.FEATS))
    ap.add_argument("--train-seeds", default=None)
    ap.add_argument("--val-seeds", default=None)
    ap.add_argument("--no-refit-val", action="store_true")
    a = ap.parse_args()
    if a.exclude_family and Path(a.out).resolve() == (io_utils.RES / "models" / "boosting").resolve():
        sys.exit("--exclude-family must not overwrite the direct-predictor models: give a separate --out")
    rng_ = lambda spec: [x for part in str(spec).split(",") for x in (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))]
    seeds = None
    if a.val_seeds:
        globals()["VAL_SEEDS"] = tuple(rng_(a.val_seeds))
    if a.train_seeds:
        seeds = set(rng_(a.train_seeds)) | set(globals()["VAL_SEEDS"])
        assert not set(rng_(a.train_seeds)) & set(globals()["VAL_SEEDS"]), "training and validation seeds overlap"
    train(a.arm, a.tag, a.out, a.exclude_family, a.feats, seeds, refit_val=not a.no_refit_val)
