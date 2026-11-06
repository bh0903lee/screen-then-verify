"""Saturation of clipped value labels."""
import argparse, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
import run_experiment as M, operators as P, io_utils

OUT = io_utils.RES / "label_saturation"


def collect(tag, arm, seed, cfg):
    band = M.get_band(M.TRAIN_BAND); groups = P.arm_groups(arm); tt, _ = M.targets_for(cfg, seed)
    t0 = time.time(); data = M.collect_value_data(groups, M.arm_id(arm), band, tt, seed, cfg)
    u = float(M.TRAIN_BAND[1:]); rows = []
    for (G, res, af, v, fam, *_) in data:
        w = M.worst(res) / u
        rows.append(dict(tag=tag, arm=arm, seed=seed, family=fam, n=G.number_of_nodes(), m=G.number_of_edges(),
                         w_state=w, sat_state=int(w <= 1.0), v_raw=v, y_clip3=min(v, 3.0),
                         y_rel=min(v / max(w, 1e-3), 1.0), af_nonzero=int(np.abs(af).sum() > 0)))
    df = pd.DataFrame(rows); OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{tag}_{arm}_s{seed}.csv"; df.to_csv(p, index=False)
    print(f"wrote {p} ({len(df)} states, {time.time()-t0:.0f}s)  sat-rate P(v>=3)={np.mean(df.v_raw >= 3):.3f}")
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="w1"); ap.add_argument("--arm", default="S_swap")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--source", default="mix7")
    ap.add_argument("--oracle-lam", default="exact", choices=["eigsh", "eigh", "exact"])
    a = ap.parse_args()
    M.set_oracle_lam(a.oracle_lam)
    cfg = M.Cfg(source=a.source, oracle_lam=a.oracle_lam); cfg.arm_weights = M.arm_weights_of(a.arm)
    collect(a.tag, a.arm, a.seed, cfg)


if __name__ == "__main__":
    main()
