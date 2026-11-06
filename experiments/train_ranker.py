"""Train edit-head graph neural rankers (learner scaling) or distil the sketch."""
import argparse, pickle, sys, time
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import numpy as np, pandas as pd
import io_utils, predictors as PR, edit_head as NN

DEPLOY_K = 32


def seeds_of(spec):
    out = []
    for part in str(spec).split(","):
        a, _, b = part.partition("-"); out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def load_states(arm, tag, seeds, target, pe, feat="L"):
    od = io_utils.RES / f"run_{tag}"; states = []
    for sd in seeds:
        fc, ff, fp = od / f"cand_{arm}_s{sd}.csv", od / f"feat_{arm}_s{sd}.npz", od / f"pools_{arm}_s{sd}.pkl"
        if not (fc.exists() and ff.exists() and fp.exists()):
            raise FileNotFoundError(f"missing pool files for {arm} seed {sd} under {od}")
        df = pd.read_csv(fc); F = np.load(ff)["F"]; pools = pickle.loads(fp.read_bytes()); X_L = PR.build_X_table(F, df, feat)
        y = (df.W1 - df.W0).to_numpy(np.float64) if target == "dW" else df[target].to_numpy(np.float64)
        i = 0
        for (ti, step), p in pools.items():
            k = len(p["ops"]); rows = df.iloc[i:i + k]
            assert (rows.target == ti).all() and (rows.step == step).all() and list(rows.pos) == list(range(k)), f"{fc}: pool/row misalignment at {(ti, step)}"
            X, ei = NN.node_inputs(int(p["n"]), p["edges"], pe=pe)
            states.append(dict(seed=sd, X=X, ei=ei, ops=p["ops"], F=X_L[i:i + k], y=y[i:i + k], W1=rows.W1.to_numpy(np.float64)))
            i += k
        assert i == len(df), f"{fc}: {len(df) - i} rows not covered by pools"
    return states


def state_metrics(pred, s):
    from scipy.stats import spearmanr
    p = pred[:DEPLOY_K]; w = s["W1"][:DEPLOY_K]; y = s["y"][:DEPLOY_K]; o = np.argsort(p, kind="stable")
    rho = spearmanr(p, y).statistic if np.std(p) > 0 and np.std(y) > 0 else np.nan
    return dict(spearman=rho, regret1=float(w[o[0]] - w.min()), regret8=float(w[o[:8]].min() - w.min()))


def evaluate(net, states, device, bs=16):
    import torch
    out = []
    net.eval()
    with torch.no_grad():
        for i in range(0, len(states), bs):
            ch = states[i:i + bs]; b = NN.collate(ch, device, None); pr = NN.forward(net, b).cpu().numpy(); c = 0
            for s in ch:
                k = len(s["ops"]); out.append(state_metrics(pr[c:c + k], s)); c += k
    d = pd.DataFrame(out)
    return dict(spearman=float(d.spearman.mean()), regret1=float(d.regret1.mean()), regret8=float(d.regret8.mean()))


def train(family, loss, tr, va, n_in, n_local, ysd, args, objectives, lr, width):
    import torch
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); dev = args.device
    net = NN.build(family, n_in, n_local, d=width).to(dev); opt = torch.optim.Adam(net.parameters(), lr=lr)
    best = {o: (np.inf, None, -1) for o in objectives}; curve = []; bad = 0
    for ep in range(1, args.epochs + 1):
        net.train(); order = rng.permutation(len(tr)); t0 = time.time(); tot = 0.0
        for i in range(0, len(order), args.batch):
            ch = [tr[j] for j in order[i:i + args.batch]]
            if family != "gin":
                ch = [dict(s, X=np.concatenate([s["X"][:, :NN.N_CHEAP], s["X"][:, NN.N_CHEAP:NN.N_CHEAP + NN.N_PE] * rng.choice([-1.0, 1.0], NN.N_PE).astype(np.float32),
                                                s["X"][:, NN.N_CHEAP + NN.N_PE:]], 1)) for s in ch]
            b = NN.collate(ch, dev, None); pr = NN.forward(net, b)
            y = torch.as_tensor(np.concatenate([s["y"] for s in ch]) / ysd, dtype=torch.float32, device=dev)
            if loss == "mse":
                L = ((pr - y) ** 2).mean()
            elif loss == "listnet":
                parts = []; c = 0
                for s in ch:
                    k = len(s["ops"]); parts.append(-(torch.softmax(-y[c:c + k], 0) * torch.log_softmax(-pr[c:c + k], 0)).sum()); c += k
                L = torch.stack(parts).mean()
            else:
                parts = []; c = 0
                for s in ch:
                    k = len(s["ops"]); ii = torch.as_tensor(rng.integers(0, k, 4 * k), device=dev); jj = torch.as_tensor(rng.integers(0, k, 4 * k), device=dev)
                    yi, yj = y[c + ii], y[c + jj]; keep = yi != yj; sgn = torch.sign(yj - yi)[keep]
                    parts.append(torch.nn.functional.softplus(-(sgn * (pr[c + jj] - pr[c + ii])[keep]))); c += k
                L = torch.cat(parts).mean() if sum(len(p_) for p_ in parts) else None
            if L is None:
                continue
            opt.zero_grad(); L.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tot += float(L)
        m = evaluate(net, va, dev); curve.append(dict(loss=loss, lr=lr, width=width, epoch=ep, train_loss=tot, sec=round(time.time() - t0, 1), **m))
        print(f"  [{family}/{loss}/lr{lr}/d{width}] epoch {ep}: {m} ({time.time() - t0:.0f}s)", flush=True)
        improved = False
        for o in objectives:
            v = -m["spearman"] if o == "spearman" else m["regret1" if o == "top1" else "regret8"]
            if v < best[o][0] - 1e-9:
                best[o] = (v, {k_: t.detach().cpu().clone() for k_, t in net.state_dict().items()}, ep); improved = True
        bad = 0 if improved else bad + 1
        if bad >= args.patience:
            break
    return best, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--pool-tag", default="r_a2pool")
    ap.add_argument("--train-seeds", required=True); ap.add_argument("--val-seeds", default="90-93")
    ap.add_argument("--family", required=True, choices=NN.FAMILIES); ap.add_argument("--target", default="dW",
                    choices=["dW", "What", "rh_spectral_gap", "rh_modularity", "rh_clustering"])
    ap.add_argument("--losses", nargs="+", default=["mse", "rank", "listnet"]); ap.add_argument("--out", required=True)
    ap.add_argument("--feat", default="L", choices=["L", "LS"])
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4]); ap.add_argument("--widths", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); args = a
    tr_s, va_s = seeds_of(a.train_seeds), seeds_of(a.val_seeds)
    assert not set(tr_s) & set(va_s), "training and validation seeds overlap"
    assert not set(tr_s + va_s) & set(range(0, 20)), "evaluation seeds 0–19 must never be read"
    pe = a.family != "gin"; t0 = time.time()
    tr = load_states(a.arm, a.pool_tag, tr_s, a.target, pe, a.feat); va = load_states(a.arm, a.pool_tag, va_s, a.target, pe, a.feat)
    Fall = np.concatenate([s["F"] for s in tr]); mu = Fall.mean(0); sd = Fall.std(0) + 1e-6
    for s in tr + va:
        s["F"] = ((s["F"] - mu) / sd).astype(np.float32)
    ysd = float(np.concatenate([s["y"] for s in tr]).std() + 1e-9)
    n_tr = int(sum(len(s["ops"]) for s in tr)); print(f"{a.arm} {a.family} {a.target}: {len(tr)} train states ({n_tr} cands), {len(va)} val states, load {time.time() - t0:.0f}s", flush=True)
    objectives = ("top1", "top8") if a.target == "dW" else ("spearman",)
    out = Path(a.out); out = out if out.is_absolute() else _REPO / out; out.mkdir(parents=True, exist_ok=True)
    allbest = {o: (np.inf, None, -1, None) for o in objectives}; curves = []
    for loss in a.losses:
        for lr in a.lrs:
            for width in a.widths:
                best, curve = train(a.family, loss, tr, va, tr[0]["X"].shape[1], Fall.shape[1], ysd, args, objectives, lr, width)
                curves += curve
                for o in objectives:
                    if best[o][0] < allbest[o][0]:
                        allbest[o] = (*best[o], (loss, lr, width))
    pd.DataFrame(curves).assign(arm=a.arm, family=a.family, target=a.target, n_train_cands=n_tr, train_seeds=a.train_seeds).to_csv(out / f"val_{a.arm}.csv", index=False)
    for o, (v, state, ep, cfg_) in allbest.items():
        loss, lr, width = cfg_
        meta = dict(family=a.family, feat=a.feat, target=a.target, objective=o, loss=loss, lr=lr, epoch=ep, val=float(v), n_in=int(tr[0]["X"].shape[1]), n_local=int(Fall.shape[1]),
                    x_mu=mu.tolist(), x_sd=sd.tolist(), y_sd=ysd, n_train_cands=n_tr, train_seeds=a.train_seeds, val_seeds=a.val_seeds, arch={"d": int(width)},
                    trained=time.strftime("%Y-%m-%d %H:%M:%S"))
        (out / f"model_{a.arm}_{o}.pkl").write_bytes(pickle.dumps(dict(state=state, meta=meta), protocol=4))
        print(f"  → {o}: {loss} lr {lr} d {width} epoch {ep} val {v:.4f}", flush=True)
    (out / f"{a.arm}.DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
