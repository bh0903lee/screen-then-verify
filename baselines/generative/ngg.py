import argparse, glob, json, os, pickle, sys, time, types
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; NGG = REPO / "external" / "NGG"
sys.path.insert(0, str(REPO / "stv")); sys.path.insert(0, str(NGG))
sys.modules.setdefault("ppgn", types.SimpleNamespace(Powerful=None))
import numpy as np, networkx as nx, torch, torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj
from autoencoder import VariationalAutoEncoder
from denoise_model import DenoiseNN, p_losses, sample as dn_sample


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)
import targets as PT

KEYS = list(PT.KEYS); COND = KEYS + ["n", "m"]; OUT = Path(os.environ.get("GEN_OUT", REPO / "results" / "generative"))
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def assert_gpu_ok():
    pass


def load_rows():
    rows = [r for f in sorted((OUT / "data").glob("shard*.pkl")) for r in pickle.load(open(f, "rb"))]
    rows.sort(key=lambda r: r["i"]); return rows


def to_data(r, n_max, spec=10):
    G = nx.Graph(); G.add_nodes_from(range(r["n"])); G.add_edges_from(map(tuple, r["edges"].tolist()))
    src = max(G.degree, key=lambda t: t[1])[0]; order = list(nx.bfs_tree(G, src).nodes())
    A = nx.to_numpy_array(G, nodelist=order); deg = A.sum(0)
    with np.errstate(divide="ignore"):
        dh = np.where(deg > 0, 1 / np.sqrt(deg), 0.0)
    L = np.eye(len(order)) - dh[:, None] * A * dh[None, :]
    w, V = np.linalg.eigh(L); V = V[:, np.argsort(w)]
    x = torch.zeros(len(order), spec + 1); x[:, 0] = torch.tensor(deg / (n_max - 1)); k = min(len(order), spec); x[:, 1:k + 1] = torch.tensor(V[:, :k])
    ei = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)
    c = torch.tensor([[r[k_] for k_ in COND]], dtype=torch.float)
    return Data(x=x, edge_index=ei, cond=c, n=len(order))


def split(rows):
    rng = np.random.default_rng(13); idx = rng.permutation(len(rows)); nv = len(rows) // 10
    return idx[nv:], idx[:nv]


POS_W = 1.0


def vae_loss(vae, data, n_max, beta=0.05):
    h = vae.encoder(data); mu, lv = vae.fc_mu(h), vae.fc_logvar(h); z = vae.reparameterize(mu, lv)
    adj = vae.decoder(z); A = to_dense_adj(data.edge_index, data.batch, max_num_nodes=n_max)
    recon = (torch.abs(adj - A) * (1.0 + (POS_W - 1.0) * A)).sum(); kld = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp())
    return recon + beta * kld, recon, kld


def build(a):
    vae = VariationalAutoEncoder(11, a.hid_enc, a.hid_dec, a.latent, 2, 3, a.n_max).to(dev)
    dn = DenoiseNN(input_dim=a.latent, hidden_dim=a.hid_dn, n_layers=3, n_cond=len(COND), d_cond=128).to(dev)
    return vae, dn


def norm_stats(rows, tr):
    C = np.array([[rows[i][k] for k in COND] for i in tr], float); return C.mean(0), C.std(0) + 1e-8


def cmd_train(a):
    rows = load_rows(); tr, va = split(rows); mu, sd = norm_stats(rows, tr)
    cache = OUT / f"ngg_data_n{a.n_max}.pt"
    if cache.exists():
        data = torch.load(cache, weights_only=False)
    else:
        data = [to_data(r, a.n_max) for r in rows]; torch.save(data, cache)
    if a.subset:
        tr = tr[: a.subset]; va = va[: max(300, a.subset // 10)]
    for d in data:
        d.cond = (d.cond - torch.tensor(mu, dtype=torch.float)) / torch.tensor(sd, dtype=torch.float)
    tl = DataLoader([data[i] for i in tr], batch_size=a.bs, shuffle=True); vl = DataLoader([data[i] for i in va], batch_size=a.bs)
    vae, dn = build(a); ck = OUT / f"ngg{a.tag_ck}"; ck.mkdir(parents=True, exist_ok=True)
    json.dump(dict(vars(a), cond=COND, mu=mu.tolist(), sd=sd.tolist(), n_train=len(tr), n_val=len(va)), open(ck / "config.json", "w"), indent=1)
    opt = torch.optim.Adam(vae.parameters(), lr=a.lr); best = np.inf; best_f1 = -1.0
    for ep in range(1, a.ep_ae + 1):
        vae.train(); s = 0
        for b in tl:
            b = b.to(dev); opt.zero_grad(); loss, _, _ = vae_loss(vae, b, a.n_max, a.beta); loss.backward(); opt.step(); s += loss.item()
        vae.eval(); v = 0
        with torch.no_grad():
            for b in vl:
                v += vae_loss(vae, b.to(dev), a.n_max, a.beta)[1].item()
        v /= len(va); f1, pe, te = val_f1(vae, [data[i] for i in va[:100]])
        print(f"[ae] ep {ep} train {s/len(tr):.2f} val_recon/graph {v:.2f} val edge-F1 {f1:.3f} (pred edges {pe:.0f} vs true {te:.0f})", flush=True)
        if f1 > best_f1 or (f1 == best_f1 and v < best):
            best_f1, best = f1, v; torch.save(vae.state_dict(), ck / "vae.pt")
    vae.load_state_dict(torch.load(ck / "vae.pt")); vae.eval()
    f1, pe, te, f1s = val_f1(vae, [data[i] for i in va[:300]], per_graph=True)
    print(f"[ae] best checkpoint: val edge-F1 mean {f1:.3f} (10th pct {np.percentile(f1s, 10):.3f}; pred edges {pe:.0f} vs true {te:.0f})", flush=True)
    if a.ep_dn == 0:
        return
    betas = linear_beta_schedule(timesteps=a.T); al = torch.cumprod(1 - betas, 0)
    s1, s2 = torch.sqrt(al), torch.sqrt(1 - al)
    opt = torch.optim.Adam(dn.parameters(), lr=a.lr); best = np.inf
    for ep in range(1, a.ep_dn + 1):
        dn.train(); s = 0
        for b in tl:
            b = b.to(dev); opt.zero_grad()
            with torch.no_grad():
                z = vae.encode(b)
            t = torch.randint(0, a.T, (z.size(0),), device=dev)
            loss = p_losses(dn, z, t, b.cond, s1, s2, loss_type="huber"); loss.backward(); opt.step(); s += loss.item() * z.size(0)
        dn.eval(); v = 0
        with torch.no_grad():
            g = torch.Generator(device=dev).manual_seed(0)
            for b in vl:
                b = b.to(dev); z = vae.encode(b); t = torch.randint(0, a.T, (z.size(0),), device=dev, generator=g)
                v += p_losses(dn, z, t, b.cond, s1, s2, noise=torch.randn(z.shape, device=dev, generator=g), loss_type="huber").item() * z.size(0)
        v /= len(va); print(f"[dn] ep {ep} train {s/len(tr):.4f} val {v:.4f}", flush=True)
        if v < best:
            best = v; torch.save(dn.state_dict(), ck / "denoiser.pt")


def load_trained():
    ck = OUT / f"ngg{os.environ.get('NGG_TAG', '')}"
    cfg = json.load(open(ck / "config.json")); a = argparse.Namespace(**cfg); vae, dn = build(a)
    vae.load_state_dict(torch.load(ck / "vae.pt", map_location=dev)); dn.load_state_dict(torch.load(ck / "denoiser.pt", map_location=dev))
    vae.eval(); dn.eval(); return a, vae, dn, np.array(cfg["mu"]), np.array(cfg["sd"])


@torch.no_grad()
def generate(a, vae, dn, mu, sd, conds, seed):
    torch.manual_seed(seed)
    c = torch.tensor((np.asarray(conds, float) - mu) / sd, dtype=torch.float, device=dev)
    z = dn_sample(dn, c, latent_dim=a.latent, timesteps=a.T, betas=linear_beta_schedule(timesteps=a.T), batch_size=c.size(0))[-1]
    adj = vae.decode_mu(z).cpu().numpy(); out = []
    for A in adj:
        G = nx.from_numpy_array(A); G.remove_nodes_from([v for v, d in dict(G.degree()).items() if d == 0])
        if G.number_of_nodes():
            G = nx.convert_node_labels_to_integers(G.subgraph(max(nx.connected_components(G), key=len)).copy())
        out.append(G)
    return out


def val_f1(vae, ds, per_graph=False, seed=0):
    f1s, pe, te = [], [], []
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []), torch.no_grad():
        torch.manual_seed(seed)
        for d in ds:
            b = next(iter(DataLoader([d], batch_size=1))).to(dev)
            adj = vae.decode_mu(vae.fc_mu(vae.encoder(b)))[0].cpu().numpy()[: d.n, : d.n]
            T_ = to_dense_adj(d.edge_index, max_num_nodes=d.n)[0].numpy(); iu = np.triu_indices(d.n, 1)
            p_, t_ = adj[iu] > .5, T_[iu] > .5; f1s.append(2 * (p_ & t_).sum() / max(1, p_.sum() + t_.sum())); pe.append(p_.sum()); te.append(t_.sum())
    vae.train(False)
    out = (float(np.mean(f1s)), float(np.mean(pe)), float(np.mean(te)))
    return out + (f1s,) if per_graph else out


def metrics(G):
    if G.number_of_nodes() < PT.MIN_N or G.number_of_edges() < 2:
        return None
    try:
        return PT.target_from(G)
    except Exception:
        return None


def cmd_check(a):
    A_, vae, dn, mu, sd = load_trained(); rows = load_rows(); tr, va = split(rows)
    sub = [rows[i] for i in va[:300]]; f1s = []
    for r in sub:
        d = to_data(r, A_.n_max); b = next(iter(DataLoader([d], batch_size=1))).to(dev)
        with torch.no_grad():
            adj = vae.decode_mu(vae.fc_mu(vae.encoder(b)))[0].cpu().numpy()[: d.n, : d.n]
        T = to_dense_adj(d.edge_index, max_num_nodes=d.n)[0].numpy(); iu = np.triu_indices(d.n, 1)
        p, t = adj[iu] > .5, T[iu] > .5; tp = (p & t).sum(); f1s.append(2 * tp / max(1, p.sum() + t.sum()))
    print(f"(3a) reconstruction edge-F1 on 300 val graphs: mean {np.mean(f1s):.3f}, 10th pct {np.percentile(f1s,10):.3f}", flush=True)
    conds = np.array([[r[k] for k in COND] for r in sub]); Gs = generate(A_, vae, dn, mu, sd, conds, seed=1)
    gm = [metrics(G) for G in Gs]; ok = [i for i, m in enumerate(gm) if m]
    rng = np.random.default_rng(2); base = [rows[i] for i in rng.choice(tr, size=len(sub))]
    print(f"(3b) valid generated graphs: {len(ok)}/{len(sub)}; mean N {np.mean([Gs[i].number_of_nodes() for i in ok]):.1f}, mean m {np.mean([Gs[i].number_of_edges() for i in ok]):.1f}")
    for k in KEYS:
        mae_g = np.mean([abs(gm[i][k] - sub[i][k]) for i in ok]); mae_b = np.mean([abs(base[i][k] - sub[i][k]) for i in ok])
        print(f"     {k:16s} MAE generated {mae_g:.4f}  random-train baseline {mae_b:.4f}  ratio {mae_g/mae_b:.2f}")
    for j, k in enumerate(KEYS):
        lo, hi = conds[:50].copy(), conds[:50].copy(); lo[:, j] -= sd[j]; hi[:, j] += sd[j]
        ml = [metrics(G) for G in generate(A_, vae, dn, mu, sd, lo, 3)]; mh = [metrics(G) for G in generate(A_, vae, dn, mu, sd, hi, 3)]
        d_ = [h[k] - l[k] for l, h in zip(ml, mh) if l and h]
        print(f"(3c) {k:16s} +1SD vs -1SD condition: mean change {np.mean(d_):+.4f} (positive in {np.mean(np.array(d_) > 0)*100:.0f}% of {len(d_)})", flush=True)


TARGET_CACHES = ("s2c", "m4")


def eval_targets(seeds, n_per_seed=28):
    per = {}
    for c in TARGET_CACHES:
        for f in glob.glob(str(REPO / "results/start_cache" / c / "targets_*.json")):
            js = json.load(open(f)); idn = js["ident"]
            if idn.get("source") == "mix7" and idn.get("oracle_lam") == "exact" and int(idn["seed"]) in seeds:
                ev = [({k: t[k] for k in KEYS}, fam) for t, fam in js["eval"]]
                sd_ = int(idn["seed"]); assert sd_ not in per or per[sd_] == ev, f"seed {sd_}: conflicting target lists across caches"
                per[sd_] = ev
    missing = sorted(set(seeds) - set(per)); assert not missing, f"no mix7 exact target file for seeds {missing}"
    out = [(sd_, ti, fam, t) for sd_ in sorted(per) for ti, (t, fam) in enumerate(per[sd_])]
    assert all(len(per[s]) == n_per_seed for s in per), {s: len(per[s]) for s in per}
    assert len({(s, ti) for s, ti, _, _ in out}) == len(out) == n_per_seed * len(set(seeds))
    return out


def cmd_sample(a):
    A_, vae, dn, mu, sd = load_trained(); seeds = [int(s) for s in a.seeds.split(",")]; T = eval_targets(seeds)
    assert T, "no exact-oracle mix7 target files for these seeds"
    res = []; t0 = time.time()
    for (sd_, ti, fam, t) in T:
        cond = np.array([[t[k] for k in KEYS] + [200, 600]] * a.k)
        Gs = generate(A_, vae, dn, mu, sd, cond, seed=sd_ * 1000 + ti)
        res.append(dict(seed=sd_, target=ti, family=fam, t=t, samples=[dict(n=G.number_of_nodes(), m=G.number_of_edges(), metrics=metrics(G),
                        edges=np.array(G.edges(), dtype=np.int16)) for G in Gs]))
    pickle.dump(res, open(OUT / f"ngg_samples_{a.tag}.pkl", "wb")); print(f"sampled {len(res)} targets x {a.k} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "check", "sample"])
    ap.add_argument("--n-max", type=int, default=200); ap.add_argument("--latent", type=int, default=32); ap.add_argument("--hid-enc", type=int, default=64)
    ap.add_argument("--hid-dec", type=int, default=256); ap.add_argument("--hid-dn", type=int, default=512); ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--ep-ae", type=int, default=200)
    ap.add_argument("--ep-dn", type=int, default=100); ap.add_argument("--k", type=int, default=64); ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--tag", default="eval"); ap.add_argument("--tag-ck", default=""); ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--beta", type=float, default=0.05); ap.add_argument("--pos-weight", type=float, default=1.0); a = ap.parse_args(); assert_gpu_ok()
    POS_W = a.pos_weight
    {"train": cmd_train, "check": cmd_check, "sample": cmd_sample}[a.cmd](a)
