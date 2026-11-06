"""Watts-Strogatz anchor targets."""
import logging
import sys
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_REPO / "stv"), str(_REPO / "experiments")]
import networkx as nx
import numpy as np
import graph_env as base

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ws_targets")
logging.getLogger("warmstart").setLevel(logging.WARNING)
RESULTS_DIR = base.RESULTS_DIR
KEYS = base.KEYS


def target_from(G):
    m = base.compute_metrics(G, KEYS)
    return {"clustering": m["clustering"], "avg_path_length": m["avg_path_length"],
            "modularity": m["modularity"], "spectral_gap": m["spectral_gap"],
            "assortativity": m["assortativity"]}


def sample_targets(n_targets, seed, kbar=6, N=200):
    rng = np.random.default_rng(seed + 1)
    out = []
    while len(out) < n_targets:
        p = float(rng.uniform(0.02, 0.5))
        G = nx.watts_strogatz_graph(N, kbar, p, seed=int(rng.integers(1_000_000_000)))
        if not nx.is_connected(G):
            G = nx.convert_node_labels_to_integers(
                G.subgraph(max(nx.connected_components(G), key=len)).copy())
        out.append(target_from(G))
    return out


