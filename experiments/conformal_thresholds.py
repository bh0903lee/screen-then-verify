"""Split-conformal verification thresholds for adaptive verification."""
import argparse, math, os, sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
import run_experiment as M, graph_env as base, io_utils

KEYS = base.KEYS; TOL = {k: float(base.TOL[k]) for k in KEYS}


def what_kappa(d, kappa):
    cols = []
    for k in KEYS:
        r = d[f"rh_{k}"].to_numpy(np.float64)
        if k == "spectral_gap":
            r = r + (kappa - 1.0) * d["dh_spectral_gap"].to_numpy(np.float64) / TOL[k]
        cols.append(np.abs(r))
    return np.max(np.vstack(cols), axis=0)


def state_scores(d, kappa):
    d = d.assign(What_k=what_kappa(d, kappa)); out = []
    for _, g in d.groupby(["seed", "target", "step"], sort=True):
        w = g.W1.to_numpy(); wh = g.What_k.to_numpy(); best = np.flatnonzero(np.isclose(w, w.min(), rtol=0, atol=1e-12))
        out.append(float(wh[best].min() - wh.min()))
    return np.asarray(out)


def tau_of(s, alpha):
    n = len(s); k = math.ceil((n + 1) * (1 - alpha))
    return float(np.sort(s)[k - 1]) if k <= n else float("inf")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="sketch_calibration__N200_seeds20-22")
    ap.add_argument("--kappa-tag", default=None)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1])
    ap.add_argument("--out", default=str(io_utils.RES / "models" / "conformal" / "tau.csv")); a = ap.parse_args()
    rows = []
    arms = sorted({f.name.split("_sketch_acc_")[1].rsplit("_s", 1)[0] for f in io_utils.RES.glob(f"run_{a.tag}_sketch_acc_S_*_s*.csv")})
    if not arms:
        sys.exit(f"no calibration tables run_{a.tag}_sketch_acc_<arm>_s<seed>.csv under {io_utils.RES}")
    for arm in arms:
        d = pd.concat([pd.read_csv(f) for f in sorted(io_utils.RES.glob(f"run_{a.tag}_sketch_acc_{arm}_s*.csv"))], ignore_index=True)
        kappa = M.resolve_lam_scale(f"auto:{a.kappa_tag or a.tag}", arm=arm); s = state_scores(d, kappa)
        for al in a.alphas:
            t = tau_of(s, al); d2 = d.assign(What_k=what_kappa(d, kappa))
            size = d2.groupby(["seed", "target", "step"]).What_k.apply(lambda x: int((x <= x.min() + t).sum())).mean()
            rows.append(dict(arm=arm, alpha=al, tau=t, n=len(s), kappa=round(kappa, 6), seeds=" ".join(map(str, sorted(d.seed.unique()))),
                             mean_set_calib=round(float(size), 3), frac_s_zero=round(float((s <= 1e-12).mean()), 4)))
            print(f"{arm} α={al}: τ={t:.4f} (n={len(s)} states, κ={kappa:.3f}), mean |S| on calibration states {size:.2f}")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False); os.replace(tmp, out); print("wrote", out)


if __name__ == "__main__":
    main()
