import argparse, json, os, pickle, sys, tempfile, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]; DG = REPO / "external" / "DiGress_guidance"
sys.path.insert(0, str(DG)); sys.path.insert(0, str(REPO / "stv")); sys.path.insert(0, str(REPO / "code" / "generative"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, networkx as nx
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import overrides as _ov
_ov.overrides = lambda f: f
from src import utils as dgu
from src.diffusion import diffusion_utils as du
from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion.extra_features import ExtraFeatures
from src.models.transformer_model import GraphTransformer
import targets as PT
from ngg import load_rows, split, eval_targets, assert_gpu_ok, metrics

KEYS = list(PT.KEYS); OUT = Path(os.environ.get("GEN_OUT", REPO / "results" / "generative")); dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
N_MAX = 200
HID_MLP = {"X": 128, "E": 64, "y": 128}
HID = {"dx": 256, "de": 64, "dy": 64, "n_head": 8, "dim_ffX": 256, "dim_ffE": 64, "dim_ffy": 256}


class Infos:
    max_n_nodes = N_MAX


def to_pyg(r, mu, sd):
    ei = torch.tensor(np.concatenate([r["edges"].T, r["edges"].T[::-1]], 1).astype(np.int64))
    y = torch.tensor([[(r[k] - m) / s for k, m, s in zip(KEYS, mu, sd)]], dtype=torch.float)
    return Data(x=torch.ones(r["n"], 1), edge_index=ei, edge_attr=torch.ones(ei.size(1), 1), y=y, num_nodes=r["n"])


def dense(b):
    ea = F.one_hot(torch.ones(b.edge_index.size(1), dtype=torch.long, device=b.x.device), 2).float()
    d, node_mask = dgu.to_dense(b.x, b.edge_index, ea, b.batch)
    d = d.mask(node_mask); return d.X, d.E, node_mask


class Diff:
    def __init__(self, T, x_marg, e_marg):
        self.T = T; self.sched = PredefinedNoiseScheduleDiscrete("cosine", T).to(dev); self.sched.alphas_bar = self.sched.alphas_bar.to(dev)
        self.trans = MarginalUniformTransition(x_marg.to(dev), e_marg.to(dev), y_classes=0)
        self.limit = dgu.PlaceHolder(X=x_marg.to(dev), E=e_marg.to(dev), y=torch.ones(0, device=dev))
        self.extra = ExtraFeatures("all", Infos())

    def noise(self, X, E, node_mask, t_int=None):
        bs = X.size(0)
        t_int = torch.randint(0, self.T + 1, (bs, 1), device=dev).float() if t_int is None else t_int
        t = t_int / self.T; ab = self.sched.get_alpha_bar(t_normalized=t); Qtb = self.trans.get_Qt_bar(ab, dev)
        s = du.sample_discrete_features(probX=X @ Qtb.X, probE=E @ Qtb.E.unsqueeze(1), node_mask=node_mask)
        z = dgu.PlaceHolder(X=F.one_hot(s.X, 1).float(), E=F.one_hot(s.E, 2).float(), y=torch.zeros(bs, 0, device=dev)).mask(node_mask)
        return {"t": t, "X_t": z.X, "E_t": z.E, "y_t": z.y, "node_mask": node_mask}

    def net_in(self, nd):
        ex = self.extra(nd); y = torch.hstack((nd["y_t"], ex.y, nd["t"])).float()
        return torch.cat((nd["X_t"], ex.X), 2).float(), torch.cat((nd["E_t"], ex.E), 3).float(), y


def build_denoiser(diff):
    X = torch.ones(1, 4, 1, device=dev); E = torch.zeros(1, 4, 4, 2, device=dev); E[..., 0] = 1; m = torch.ones(1, 4, dtype=torch.bool, device=dev)
    nd = {"t": torch.zeros(1, 1, device=dev), "X_t": X, "E_t": E, "y_t": torch.zeros(1, 0, device=dev), "node_mask": m}
    Xi, Ei, yi = diff.net_in(nd)
    return GraphTransformer(8, {"X": Xi.size(-1), "E": Ei.size(-1), "y": yi.size(-1)}, HID_MLP, HID, {"X": 1, "E": 2, "y": 0},
                            nn.ReLU(), nn.ReLU()).to(dev)


def build_regressor(diff=None, feat=False):
    if not feat:
        return GraphTransformer(5, {"X": 1, "E": 2, "y": 1}, HID_MLP, HID, {"X": 0, "E": 0, "y": len(KEYS)}, nn.ReLU(), nn.ReLU()).to(dev)
    Xi, Ei, yi = diff.net_in(_dummy_nd())
    return GraphTransformer(5, {"X": Xi.size(-1), "E": Ei.size(-1), "y": yi.size(-1)}, HID_MLP, HID, {"X": 0, "E": 0, "y": len(KEYS)},
                            nn.ReLU(), nn.ReLU()).to(dev)


def _dummy_nd():
    X = torch.ones(1, 4, 1, device=dev); E = torch.zeros(1, 4, 4, 2, device=dev); E[..., 0] = 1; m = torch.ones(1, 4, dtype=torch.bool, device=dev)
    return {"t": torch.zeros(1, 1, device=dev), "X_t": X, "E_t": E, "y_t": torch.zeros(1, 0, device=dev), "node_mask": m}


def reg_forward(reg, X_t, E_t, t, m, ex=None):
    if not getattr(reg, "feat", False):
        return reg(X_t, E_t, t, m).y
    y_t = torch.zeros(X_t.size(0), 0, device=dev)
    return reg(torch.cat((X_t, ex.X.detach()), 2).float(), torch.cat((E_t, ex.E.detach()), 3).float(),
               torch.hstack((y_t, ex.y.detach(), t)).float(), m).y


def setup(a, rows=None):
    rows = rows or load_rows(); tr, va = split(rows)
    mu = np.array([np.mean([rows[i][k] for i in tr]) for k in KEYS]); sd = np.array([np.std([rows[i][k] for i in tr]) for k in KEYS]) + 1e-8
    dens = np.mean([2 * rows[i]["m"] / (rows[i]["n"] * (rows[i]["n"] - 1)) for i in tr])
    x_marg = torch.tensor([1.0]); e_marg = torch.tensor([1 - dens, dens], dtype=torch.float)
    return rows, tr, va, mu, sd, x_marg, e_marg


def ce_loss(pred, X, E):
    tX, tE = X.reshape(-1, X.size(-1)), E.reshape(-1, E.size(-1)); pX, pE = pred.X.reshape(-1, pred.X.size(-1)), pred.E.reshape(-1, pred.E.size(-1))
    mX, mE = (tX != 0).any(-1), (tE != 0).any(-1)
    lX = F.cross_entropy(pX[mX], tX[mX].argmax(-1)) if mX.any() else 0.0
    return lX + 5.0 * F.cross_entropy(pE[mE], tE[mE].argmax(-1))


def cmd_train(a, which):
    rows, tr, va, mu, sd, xm, em = setup(a); diff = Diff(a.T, xm, em); ck = OUT / "digress"; ck.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(a), mu=mu.tolist(), sd=sd.tolist(), e_marg=em.tolist()), open(ck / f"config_{which}.json", "w"), indent=1)
    tl = DataLoader([to_pyg(rows[i], mu, sd) for i in tr], batch_size=a.bs, shuffle=True, drop_last=True)
    vl = DataLoader([to_pyg(rows[i], mu, sd) for i in va[: a.n_val]], batch_size=a.bs)
    net = build_denoiser(diff) if which == "denoiser" else build_regressor(diff, feat=which == "regressor_feat")
    net.feat = which == "regressor_feat"
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, amsgrad=True, weight_decay=1e-12); best = np.inf; step = 0; t0 = time.time()

    def loss_of(b, train):
        X, E, m = dense(b.to(dev)); nd = diff.noise(X, E, m)
        if which == "denoiser":
            Xi, Ei, yi = diff.net_in(nd); return ce_loss(net(Xi, Ei, yi, m), X, E)
        ex = diff.extra(nd) if net.feat else None
        return F.mse_loss(reg_forward(net, nd["X_t"], nd["E_t"], nd["t"], m, ex), b.y)

    for ep in range(1, a.epochs + 1):
        net.train()
        for b in tl:
            opt.zero_grad(); loss = loss_of(b, True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
            if step % 200 == 0:
                print(f"[{which}] ep {ep} step {step} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
            if a.max_steps and step >= a.max_steps:
                break
        net.eval()
        v = validate(loss_of, vl)
        print(f"[{which}] ep {ep} val {v:.4f}", flush=True)
        if v < best:
            best = v; torch.save(net.state_dict(), ck / f"{which}.pt")
        if a.max_steps and step >= a.max_steps:
            break


def validate(loss_of, vl, seed=0):
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        with torch.no_grad():
            return float(np.mean([loss_of(b, False).item() for b in vl]))


def load(which, diff):
    net = build_denoiser(diff) if which == "denoiser" else build_regressor(diff, feat=which == "regressor_feat")
    net.load_state_dict(torch.load(OUT / "digress" / f"{which}.pt", map_location=dev)); net.eval(); net.feat = which == "regressor_feat"; return net


def p_zs_given_zt(diff, den, reg, s, t, X_t, E_t, m, target, lam):
    bs, n, _ = X_t.shape
    beta_t = diff.sched(t_normalized=t); ab_s = diff.sched.get_alpha_bar(t_normalized=s); ab_t = diff.sched.get_alpha_bar(t_normalized=t)
    Qtb, Qsb, Qt = diff.trans.get_Qt_bar(ab_t, dev), diff.trans.get_Qt_bar(ab_s, dev), diff.trans.get_Qt(beta_t, dev)
    nd = {"t": t, "X_t": X_t, "E_t": E_t, "y_t": torch.zeros(bs, 0, device=dev), "node_mask": m}
    with torch.no_grad():
        ex = diff.extra(nd)
        Xi = torch.cat((nd["X_t"], ex.X), 2).float(); Ei = torch.cat((nd["E_t"], ex.E), 3).float()
        yi = torch.hstack((nd["y_t"], ex.y, nd["t"])).float(); pred = den(Xi, Ei, yi, m)
    pX, pE = F.softmax(pred.X, -1), F.softmax(pred.E, -1)
    psX = du.compute_batched_over0_posterior_distribution(X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X)
    psE = du.compute_batched_over0_posterior_distribution(X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E)
    uX = (pX.unsqueeze(-1) * psX).sum(2); uX[uX.sum(-1) == 0] = 1e-5; prob_X = uX / uX.sum(-1, keepdim=True)
    pE = pE.reshape(bs, -1, pE.shape[-1]); uE = (pE.unsqueeze(-1) * psE).sum(-2); uE[uE.sum(-1) == 0] = 1e-5
    prob_E = (uE / uE.sum(-1, keepdim=True)).reshape(bs, n, n, -1)
    if lam > 0:
        with torch.enable_grad():
            x_in = X_t.float().detach().requires_grad_(True); e_in = E_t.float().detach().requires_grad_(True)
            mse = ((reg_forward(reg, x_in, e_in, t, m, ex) - target.expand(bs, -1)) ** 2).mean(-1).sum()
            gx = torch.autograd.grad(mse, x_in, retain_graph=True)[0]; ge = torch.autograd.grad(mse, e_in)[0]
        xm = m.unsqueeze(-1).float(); diag = (~torch.eye(n, dtype=torch.bool, device=dev)).float().unsqueeze(0).unsqueeze(-1)
        ge = ge * xm.unsqueeze(2) * xm.unsqueeze(1) * diag; ge = 0.5 * (ge + ge.transpose(1, 2)); gx = gx * xm
        px = torch.softmax(-lam * gx, -1) * prob_X; px[px.sum(-1) == 0] = 1e-7; prob_X = px / px.sum(-1, keepdim=True)
        pe = torch.softmax(-lam * ge, -1) * prob_E; pe[pe.sum(-1) == 0] = 1e-7; prob_E = pe / pe.sum(-1, keepdim=True)
    smp = du.sample_discrete_features(prob_X, prob_E, node_mask=m)
    return F.one_hot(smp.X, 1).float() * m.unsqueeze(-1), F.one_hot(smp.E, 2).float()


def generate(diff, den, reg, target_z, k, lam, seed, bs=16):
    tgt = torch.tensor(target_z, dtype=torch.float, device=dev).unsqueeze(0); A = []
    for c, k0 in enumerate(range(0, k, bs)):
        kc = min(bs, k - k0); torch.manual_seed(seed * 1000 + c); m = torch.ones(kc, N_MAX, dtype=torch.bool, device=dev)
        z = du.sample_discrete_feature_noise(limit_dist=diff.limit, node_mask=m); X, E = z.X, z.E
        for s_int in reversed(range(diff.T)):
            s = torch.full((kc, 1), s_int / diff.T, device=dev); t = s + 1 / diff.T
            X, E = p_zs_given_zt(diff, den, reg, s, t, X, E, m, tgt, lam)
        A.append(E.argmax(-1).cpu().numpy())
    A = np.concatenate(A); assert len(A) == k; out = []
    for Ai in A:
        G = nx.from_numpy_array(np.triu(Ai, 1) + np.triu(Ai, 1).T); G.remove_nodes_from([v for v, d in dict(G.degree()).items() if d == 0])
        if G.number_of_nodes():
            G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
        out.append(G)
    return out


def atomic_pickle_dump(value, path):
    path = Path(path); tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            pickle.dump(value, f)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def lineage_digest(a, T_):
    import hashlib, subprocess
    def sha(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    d = OUT / "digress"; files = ["denoiser.pt", "config_denoiser.json"] + ([f"{a.reg}.pt"] if a.lam > 0 else [])
    tg = hashlib.sha256(json.dumps([[s_, ti, fam, [float(t[k]) for k in KEYS]] for (s_, ti, fam, t) in T_], sort_keys=True).encode()).hexdigest()
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent), timeout=10).stdout.strip()
    except Exception:
        rev = ""
    return dict(compared=dict(files={f: sha(d / f) for f in files}, targets=tg),
                info=dict(git=rev, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", started=time.strftime("%F_%T", time.gmtime())))


def cmd_sample(a):
    with open(OUT / "digress" / "config_denoiser.json") as f:
        cfg = json.load(f)
    mu, sd = np.array(cfg["mu"]), np.array(cfg["sd"])
    diff = Diff(cfg["T"], torch.tensor([1.0]), torch.tensor(cfg["e_marg"])); den = load("denoiser", diff); reg = load(a.reg, diff) if a.lam > 0 else None
    seeds = [int(x) for x in a.seeds.split(",")]; T_ = eval_targets(seeds)
    res = []; t0 = time.time(); part = OUT / f"digress_samples_{a.tag}_partial.pkl"
    meta = dict(seeds=seeds, k=a.k, lam=a.lam, reg=a.reg, bs_sample=a.bs_sample, swap=bool(a.swap))
    lin = lineage_digest(a, T_); lin["segments"] = [dict(first_target=0, **lin["info"])]
    if a.resume and part.exists():
        old = pickle.load(open(part, "rb")); assert old["meta"] == meta, (old["meta"], meta); res = old["results"]
        if "lineage" in old:
            assert old["lineage"]["compared"] == lin["compared"], ("lineage mismatch — refusing to mix models in one sample file", old["lineage"]["compared"], lin["compared"])
            lin["segments"] = old["lineage"]["segments"] + [dict(first_target=len(res), **lin["info"])]
        else:
            lin["segments"] = [dict(first_target=0, legacy_without_lineage=True, n_targets=len(res)), dict(first_target=len(res), **lin["info"])]
            print(f"resume: partial has no lineage record — its {len(res)} targets are adopted as 'legacy_without_lineage' (not verified against the current digests)", flush=True)
        print(f"resume: {len(res)} targets from {part.name}", flush=True)
    done = {(r["seed"], r["target"]) for r in res}
    by_seed = {s_: [x for x in T_ if x[0] == s_] for s_ in seeds}
    for (sd_, ti, fam, t) in T_[: a.limit or None]:
        if (sd_, ti) in done:
            continue
        cond = t
        if a.swap:
            cj = (ti + 14) % len(by_seed[sd_]); cond = by_seed[sd_][cj][3]; assert by_seed[sd_][cj][1] == cj and cond != t
        Gs = generate(diff, den, reg, [(cond[k] - m_) / s_ for k, m_, s_ in zip(KEYS, mu, sd)], a.k, a.lam, sd_ * 1000 + ti, bs=a.bs_sample)
        res.append(dict(seed=sd_, target=ti, family=fam, t=t, cond=cond, samples=[dict(n=G.number_of_nodes(), m=G.number_of_edges(), metrics=metrics(G),
                        edges=np.array(G.edges(), dtype=np.int16)) for G in Gs]))
        print(f"target {sd_}/{ti} done ({time.time()-t0:.0f}s)", flush=True)
        atomic_pickle_dump(dict(meta=meta, lineage=lin, results=res), part)
    assert len(res) == len(T_[: a.limit or None]) > 0
    atomic_pickle_dump(dict(meta=dict(**meta, n_targets=len(res)), lineage=lin, results=res), OUT / f"digress_samples_{a.tag}.pkl")
    part.unlink(missing_ok=True)


def load_reg(which, diff, ckpt=None):
    net = build_regressor(diff, feat=which == "regressor_feat")
    net.load_state_dict(torch.load(ckpt or OUT / "digress" / f"{which}.pt", map_location=dev)); net.eval(); net.feat = which == "regressor_feat"; return net


def val_mse_by_t(reg, diff, vl, t_grid, seed=0):
    out = {}
    for t in t_grid:
        se = []
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []), torch.no_grad():
            torch.manual_seed(seed + int(t))
            for b in vl:
                b = b.to(dev); X, E, m = dense(b); nd = diff.noise(X, E, m, t_int=torch.full((X.size(0), 1), float(t), device=dev))
                ex = diff.extra(nd) if getattr(reg, "feat", False) else None
                se.append(((reg_forward(reg, nd["X_t"], nd["E_t"], nd["t"], m, ex) - b.y) ** 2).cpu().numpy())
        se = np.concatenate(se); out[int(t)] = (float(se.mean()), [float(x) for x in se.mean(0)])
    return out


def cmd_valmse(a):
    cfg = json.load(open(OUT / "digress" / "config_denoiser.json")); rows, tr, va, mu, sd, xm, em = setup(a)
    assert a.T == cfg["T"], (a.T, cfg["T"]); diff = Diff(a.T, xm, em); reg = load_reg(a.reg, diff, a.ckpt)
    vl = DataLoader([to_pyg(rows[i], mu, sd) for i in va[: a.n_val]], batch_size=a.bs)
    t_grid = [int(x) for x in a.t_grid.split(",")]; r = val_mse_by_t(reg, diff, vl, t_grid)
    import hashlib; ck = a.ckpt or str(OUT / "digress" / f"{a.reg}.pt"); h = hashlib.sha256(open(ck, "rb").read()).hexdigest()[:16]
    res = dict(reg=a.reg, ckpt=ck, ckpt_sha256_16=h, n_val=len(vl.dataset), bs=a.bs, keys=KEYS, mse={t: r[t][0] for t in t_grid}, mse_by_key={t: r[t][1] for t in t_grid})
    for t in t_grid:
        print(f"[val-mse {a.reg}] t={t:3d} MSE {r[t][0]:.3f} per-key " + " ".join(f"{x:.2f}" for x in r[t][1]), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return res


def cmd_gradscale(a):
    cfg = json.load(open(OUT / "digress" / "config_denoiser.json")); rows, tr, va, mu, sd, xm, em = setup(a)
    assert a.T == cfg["T"]; diff = Diff(a.T, xm, em); reg = load_reg(a.reg, diff, a.ckpt)
    vl = DataLoader([to_pyg(rows[i], mu, sd) for i in va[: a.n_val]], batch_size=a.bs); out = {}
    for t_ in [int(x) for x in a.t_grid.split(",")]:
        q = []
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
            torch.manual_seed(7 + t_)
            for b in vl:
                b = b.to(dev); X, E, m = dense(b); nd = diff.noise(X, E, m, t_int=torch.full((X.size(0), 1), float(t_), device=dev))
                with torch.no_grad():
                    ex = diff.extra(nd) if getattr(reg, "feat", False) else None
                bs, n = X.size(0), X.size(1)
                with torch.enable_grad():
                    e_in = nd["E_t"].float().detach().requires_grad_(True)
                    mse = ((reg_forward(reg, nd["X_t"], e_in, nd["t"], m, ex) - b.y) ** 2).mean(-1).sum()
                    ge = torch.autograd.grad(mse, e_in)[0]
                xm_ = m.unsqueeze(-1).float(); diag = (~torch.eye(n, dtype=torch.bool, device=dev)).float().unsqueeze(0).unsqueeze(-1)
                ge = ge * xm_.unsqueeze(2) * xm_.unsqueeze(1) * diag; ge = 0.5 * (ge + ge.transpose(1, 2))
                valid = (m.unsqueeze(2) & m.unsqueeze(1)).unsqueeze(-1).expand_as(ge) & (diag > 0)
                q.append(ge[valid].abs().detach().cpu().numpy())
        q = np.concatenate(q); out[t_] = dict(p50=float(np.median(q)), p90=float(np.percentile(q, 90)), p99=float(np.percentile(q, 99)), max=float(q.max()), n=int(q.size))
        print(f"[grad-scale {a.reg}] t={t_:3d} |g_E| p50 {out[t_]['p50']:.2e} p90 {out[t_]['p90']:.2e} p99 {out[t_]['p99']:.2e} max {out[t_]['max']:.2e}", flush=True)
    if a.out:
        json.dump(dict(reg=a.reg, ckpt=a.ckpt, n_val=len(vl.dataset), grad=out), open(a.out, "w"), indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train-denoiser", "train-regressor", "train-regressor_feat", "sample", "val-mse", "grad-scale"])
    ap.add_argument("--T", type=int, default=500); ap.add_argument("--bs", type=int, default=12); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--n-val", type=int, default=600)
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--lam", type=float, default=0.0); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default="eval")
    ap.add_argument("--reg", default=None, choices=["regressor", "regressor_feat"]); ap.add_argument("--bs-sample", type=int, default=16)
    ap.add_argument("--swap", action="store_true"); ap.add_argument("--ckpt", default=None); ap.add_argument("--t-grid", default="0,25,50,100,200,300,400,500")
    ap.add_argument("--out", default=None); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.cmd == "sample" and a.lam > 0 and a.reg is None:
        ap.error("--lam > 0 needs --reg {regressor, regressor_feat}")
    if a.cmd == "sample" and a.swap and a.lam == 0:
        ap.error("--swap is a guidance gauge: needs --lam > 0")
    if a.cmd in ("val-mse", "grad-scale") and a.reg is None:
        ap.error(f"{a.cmd} needs --reg")
    assert_gpu_ok()
    if a.cmd == "sample":
        cmd_sample(a)
    elif a.cmd == "val-mse":
        cmd_valmse(a)
    elif a.cmd == "grad-scale":
        cmd_gradscale(a)
    else:
        cmd_train(a, a.cmd.split("-", 1)[1])
