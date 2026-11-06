import argparse, os, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stv")); sys.path.insert(0, str(REPO / "stv"))
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import targets as PT

SEED0 = 100000


def one(i):
    fam = PT.FAMILIES[i % len(PT.FAMILIES)]
    rng = np.random.default_rng(SEED0 + i)
    t = PT.sample_one(fam, rng)
    if t is None:
        return None
    return dict(i=i, family=fam, n=t["_n_src"], m=len(t["_graph_edges"]), edges=np.asarray(t["_graph_edges"], dtype=np.int16),
                **{k: t[k] for k in PT.KEYS})


def shard(args):
    w, idx, out = args
    rows = [r for r in map(one, idx) if r is not None]
    tmp = Path(out) / f".shard{w:02d}.tmp"; pickle.dump(rows, open(tmp, "wb")); os.replace(tmp, Path(out) / f"shard{w:02d}.pkl")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--start", type=int, default=0); ap.add_argument("--count", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--out", default=str(REPO / "results/generative/data")); a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True); idx = list(range(a.start, a.start + a.count))
    parts = [(w, idx[w::a.workers], a.out) for w in range(a.workers)]
    t0 = time.time()
    with Pool(a.workers) as p:
        n = p.map(shard, parts)
    print(f"wrote {sum(n)} graphs in {time.time()-t0:.0f}s ->", a.out, "keys", PT.KEYS)
