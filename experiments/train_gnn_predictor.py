"""Train the GCN direct predictor on candidate pools."""
import argparse, pickle, shutil, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd, networkx as nx, torch
import io_utils, operators as P, predictors as PR
from train_direct_predictor import load as load_table, state_scores, VAL_SEEDS, DEPLOY_K

EXP = "run"
CONFIGS = ((32, 64), (64, 64))
MAX_EP, LR, BATCH = 15, 1e-3, 64
torch.set_num_threads(1)


def load_pools(arm, tag):
    od = io_utils.RES / f"{EXP}_{tag}"; pools = {}
    for f in sorted(od.glob(f"pools_{arm}_s*.pkl")):
        sd = int(f.stem.rsplit("_s", 1)[1]); pools[sd] = pickle.loads(f.read_bytes())
    if not pools:
        sys.exit(f"no pools for {arm} under {od} (run candidate_pool.py --save-pools)")
    return pools


def pool_graph(rec):
    G = nx.Graph(); G.add_nodes_from(range(int(rec["n"]))); G.add_edges_from(map(tuple, np.asarray(rec["edges"]).tolist())); return G


def precompute(df, pools):
    keys = list(zip(df.seed.values, df.target.values, df.step.values)); X = [None] * len(df); gidx = {}; graphs = []
    for i, (sd, ti, st) in enumerate(keys):
        k = (sd, int(ti), int(st))
        if k not in gidx:
            rec = pools[sd][(int(ti), int(st))]; gidx[k] = len(graphs); graphs.append((pool_graph(rec), rec["ops"], rec["n"]))
    pos_in_pool = df.groupby(["seed", "target", "step"]).cumcount().values
    assert (pos_in_pool == df.pos.values).all(), "candidate rows must be in pool (perm) order — pools_*.pkl ops are stored in that order"
    for i, (sd, ti, st) in enumerate(keys):
        G, ops, n = graphs[gidx[(sd, int(ti), int(st))]]; op = ops[int(pos_in_pool[i])]
        assert op[0] == df.op.values[i], f"row {i}: pool op {op[0]} vs csv {df.op.values[i]}"
        P.apply_op(G, op)
        try:
            X[i] = PR.gnet_tensors(G, n)[1]
        finally:
            P.revert_op(G, op)
    return graphs, gidx, pos_in_pool, keys, X


def an_of(G, n, op):
    P.apply_op(G, op)
    try:
        return PR.gnet_adj(G, n)
    finally:
        P.revert_op(G, op)


def check_same_pools(df, arm, ref_tag="training_pools__direct_predictors"):
    od = io_utils.RES / f"{EXP}_{ref_tag}"; fs = sorted(od.glob(f"cand_{arm}_s*.csv"))
    if not fs:
        print(f"  [check] no {ref_tag} reference for {arm} — equality check skipped", flush=True); return False
    ref = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    key = ["seed", "target", "step", "pos"]; m = df.merge(ref, on=key, suffixes=("", "_ref"), how="outer", indicator=True)
    assert (m._merge == "both").all(), f"candidate pool differs from the reference pool: {int((m._merge != 'both').sum())} candidate rows differ in (seed, target, step, pos)"
    assert (m.op == m.op_ref).all() and np.allclose(m.W1, m.W1_ref, atol=1e-6) and np.allclose(m.What, m.What_ref, atol=1e-6) and np.allclose(m.W0, m.W0_ref, atol=1e-6), \
        "candidate labels differ from the reference pool"
    print(f"  [check] candidate pool matches the reference pool: {len(df)} rows identical (op, W0, W1, W_hat)", flush=True); return True


def run_epoch(net, opt, idx, data, train=True):
    graphs, gidx, pos, keys, X, Xl, y = data; idx = np.asarray(idx)
    order = np.random.default_rng(int(len(idx)) + int(np.random.randint(1 << 30))).permutation(idx) if train else idx
    net.train(train); loss_sum = 0.0; out = np.zeros(len(idx), np.float64)
    for b0 in range(0, len(order), BATCH):
        batch = order[b0: b0 + BATCH]; preds = []
        for i in batch:
            sd, ti, st = keys[i]; G, ops, n = graphs[gidx[(sd, int(ti), int(st))]]
            preds.append(net(an_of(G, n, ops[int(pos[i])]), X[i], torch.from_numpy(Xl[i])))
        pv = torch.stack(preds); tv = torch.from_numpy(y[batch])
        if train:
            loss = torch.mean((pv - tv) ** 2); opt.zero_grad(); loss.backward(); opt.step(); loss_sum += float(loss) * len(batch)
        else:
            out[b0: b0 + len(batch)] = pv.detach().numpy()
    return loss_sum / max(len(order), 1), (None if train else out)


def train(arm, tag, out, max_ep=MAX_EP, configs=CONFIGS, m3_dir=None, exclude_family=None):
    torch.manual_seed(0); np.random.seed(0)
    df, F = load_table(arm, tag); pools = load_pools(arm, tag); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    same_as_ref = check_same_pools(df, arm)
    if exclude_family:
        assert exclude_family in set(df.family), f"family {exclude_family!r} not in the table ({sorted(set(df.family))})"
        keep = (df.family != exclude_family).values; df, F = df[keep].reset_index(drop=True), np.asarray(F)[keep]
        print(f"  LOFO: excluded family {exclude_family}: {int((~keep).sum())} rows dropped, {len(df)} kept", flush=True)
    y = (df.W1 - df.W0).values.astype(np.float32); r0 = df[[c for c in df.columns if c.startswith("r0_")]].values.astype(np.float32)
    Xl = np.hstack([np.asarray(F, np.float32), r0, np.abs(r0)]).astype(np.float32)
    tr = np.where(~df.seed.isin(VAL_SEEDS).values)[0]; va = np.where(df.seed.isin(VAL_SEEDS).values & (df.pos.values < DEPLOY_K))[0]
    assert len(tr) > 1000 and len(va) > 200, f"too little data: train {len(tr)}, val {len(va)}"
    mu, sd = Xl[tr].mean(0), Xl[tr].std(0) + 1e-6; Xs = ((Xl - mu) / sd).astype(np.float32); ysd = float(y[tr].std() + 1e-6)
    t0 = time.time(); graphs, gidx, pos, keys, X = precompute(df, pools); print(f"  {arm}: {len(df)} candidates, {len(graphs)} pools, features {time.time()-t0:.0f}s", flush=True)
    data = (graphs, gidx, pos, keys, X, Xs, (y / ysd).astype(np.float32)); dv = df.iloc[va].reset_index(drop=True)
    report = []; best = {"top1": None, "top8": None}
    for emb, hid in configs:
        torch.manual_seed(0); net = PR.make_gnet(Xs.shape[1], emb, hid); opt = torch.optim.Adam(net.parameters(), lr=LR); name = f"gnet_e{emb}_h{hid}"
        for ep in range(1, max_ep + 1):
            t1 = time.time(); tl, _ = run_epoch(net, opt, tr, data, True)
            with torch.no_grad():
                _, pv = run_epoch(net, None, va, data, False)
            sc = state_scores(dv, pv); rec = dict(arm=arm, feat=PR.GFEAT, model=f"{name}_ep{ep}", n_train=len(tr), n_val=len(va), fit_s=round(time.time() - t1, 1),
                                              val_rmse=float(np.sqrt(np.mean((pv * ysd - y[va]) ** 2))), **{f"val_{k}": round(float(v), 4) for k, v in sc.items()})
            report.append(rec)
            keyd = {"top1": (rec.get("val_regret1", 9e9), -rec.get("val_hit1", 0.0)), "top8": (rec.get("val_regret8", 9e9), -rec.get("val_recall8", 0.0))}
            for obj, key in keyd.items():
                if best[obj] is None or key < best[obj][0]:
                    best[obj] = (key, name, (emb, hid), ep)
            print(f"  {arm} {name} ep{ep}: train mse {tl:.4f}, val regret8 {rec.get('val_regret8')}, recall8 {rec.get('val_recall8')}, spearman {rec.get('val_spearman')} ({rec['fit_s']}s)", flush=True)
    for ref, sc_ in (("ref_sketch", dv.What.values), ("ref_random", np.random.default_rng(0).random(len(dv)))):
        report.append(dict(arm=arm, feat=PR.GFEAT, model=ref, n_train=0, n_val=len(va), fit_s=0.0, val_rmse=np.nan, **{f"val_{k}": round(float(v), 4) for k, v in state_scores(dv, sc_).items()}))
    allidx = np.arange(len(df))
    for obj in ("top1", "top8"):
        _, name, (emb, hid), ep = best[obj]; torch.manual_seed(0); net = PR.make_gnet(Xs.shape[1], emb, hid); opt = torch.optim.Adam(net.parameters(), lr=LR); t1 = time.time()
        for _ in range(ep):
            run_epoch(net, opt, allidx, data, True)
        meta = dict(selected=f"{name}_ep{ep}", objective=obj, deploy_k=DEPLOY_K, n_fit=int(len(df)), n_local=int(Xs.shape[1]), emb=emb, hidden=hid, epochs=ep, same_pools_as_reference=bool(same_as_ref),
                    x_mu=mu.tolist(), x_sd=sd.tolist(), y_sd=ysd, fit_s=round(time.time() - t1, 1), label_oracle_calls=int(len(df)), val_seeds=list(VAL_SEEDS), tag=tag, exclude_family=exclude_family,
                    trained=time.strftime("%Y-%m-%d %H:%M:%S"))
        state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
        PR.model_path(out, arm, PR.GFEAT, obj).write_bytes(pickle.dumps(dict(state=state, feat=PR.GFEAT, arm=arm, meta=meta), protocol=4))
        print(f"  → {arm} G [{obj}]: {meta['selected']} (n_fit {len(df)}, {Xs.shape[1]} local features)", flush=True)
    m3 = Path(m3_dir) if m3_dir else io_utils.RES / "models" / "boosting"
    for feat in ("L",):
        for obj in ("top1", "top8"):
            src = PR.model_path(m3, arm, feat, obj); assert src.exists(), f"direct-predictor model {src} missing (the GNN predictor reuses the direct-predictor models)"
            shutil.copy2(src, PR.model_path(out, arm, feat, obj))
    pd.DataFrame(report).to_csv(out / f"val_{arm}.csv", index=False)
    (out / f"train_{arm}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", required=True); ap.add_argument("--tag", default="training_pools__gcn_predictor"); ap.add_argument("--out", default=str(io_utils.RES / "models" / "gcn"))
    ap.add_argument("--max-ep", type=int, default=MAX_EP); ap.add_argument("--m3-dir", default=None)
    ap.add_argument("--exclude-family", default=None)
    a = ap.parse_args()
    if a.exclude_family:
        if Path(a.out).resolve() == (io_utils.RES / "models" / "gcn").resolve():
            sys.exit("--exclude-family must not overwrite the GNN models: give a separate --out")
        if not a.m3_dir:
            sys.exit("--exclude-family needs --m3-dir pointing at direct-predictor models trained without the same family")
    train(a.arm, a.tag, a.out, max_ep=a.max_ep, m3_dir=a.m3_dir, exclude_family=a.exclude_family)
