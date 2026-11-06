from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from ..constraint_dag import ConstraintSpec

logger = logging.getLogger(__name__)


def _sample_powerlaw_degree_sequence(n: int, gamma: float, k_min: int = 1) -> list[int]:
    k_max = max(k_min, int(n ** (1 / (gamma - 1))) + 1)
    ks = np.arange(k_min, k_max + 1, dtype=float)
    probs = ks ** (-gamma)
    probs /= probs.sum()

    rng = np.random.default_rng()
    seq = rng.choice(ks.astype(int), size=n, p=probs)

    if seq.sum() % 2 != 0:
        seq[rng.integers(0, n)] += 1

    seq = np.clip(seq, k_min, n - 1)
    return seq.tolist()


def _make_graphical(degree_seq: list[int]) -> list[int]:
    seq = sorted(degree_seq, reverse=True)
    n = len(seq)

    for _ in range(100):
        if is_graphical(seq):
            break
        seq[0] = min(seq[0], n - 1)
        seq = sorted(seq, reverse=True)
        if sum(seq) % 2 != 0:
            seq[-1] = max(0, seq[-1] - 1)
        seq = sorted(seq, reverse=True)

    return seq


def is_graphical(seq: list[int]) -> bool:
    try:
        return nx.is_graphical(seq)
    except Exception:
        return False


def configuration_model_simple(degree_seq: list[int]) -> nx.Graph:
    G = nx.configuration_model(degree_seq, seed=None)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


class Stage1DegreeSequence:

    def __init__(
        self,
        spec: "ConstraintSpec",
        max_rewire_attempts: int = 1000,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.max_rewire_attempts = max_rewire_attempts
        self.verbose = verbose

    def _ws_clustering_estimate(self, k: int, p: float) -> float:
        if k <= 2:
            return 0.0
        return 3 * (k - 2) / (4 * (k - 1)) * (1 - p) ** 3

    def run(self) -> nx.Graph:
        n = self.spec.n_nodes
        m = self.spec.n_edges
        avg_deg = 2 * m / n

        if getattr(self.spec, "degree_seq", None) is not None:
            seq = list(self.spec.degree_seq)
            if len(seq) != n:
                raise ValueError(f"degree_seq length {len(seq)} != n_nodes {n}")
            if sum(seq) % 2 != 0 or not nx.is_graphical(seq):
                raise ValueError("degree_seq is not graphical (even sum + Erdős–Gallai required)")
            G = nx.havel_hakimi_graph(seq)
            n_swap = max(10 * G.number_of_edges(), 100)
            try:
                nx.double_edge_swap(G, nswap=n_swap, max_tries=n_swap * 20)
            except (nx.NetworkXError, nx.NetworkXAlgorithmError):
                pass
            G = nx.convert_node_labels_to_integers(G)
            if self.verbose:
                logger.info(f"Stage 1: explicit degree_seq (N={n}, exact degrees, {G.number_of_edges()} edges)")
            return G

        if self.spec.degree_gamma is not None:
            gamma = self.spec.degree_gamma
            logger.info(f"Stage 1: power-law degree sequence (γ={gamma}, N={n})")
            seq = _sample_powerlaw_degree_sequence(n, gamma)
            seq = _make_graphical(seq)

            if sum(seq) // 2 != m:
                scale = 2 * m / max(1, sum(seq))
                seq = [max(1, int(d * scale)) for d in seq]
                if sum(seq) % 2 != 0:
                    seq[0] += 1
                seq = _make_graphical(seq)

            G = configuration_model_simple(seq)

        elif (self.spec.clustering is not None and self.spec.clustering > 0.12):
            k_ws = max(2, int(avg_deg))
            if k_ws % 2 != 0:
                k_ws += 1
            c_ws_0 = self._ws_clustering_estimate(k_ws, 0.0)
            target_c = self.spec.clustering
            if c_ws_0 > 0:
                ratio = target_c / c_ws_0
                p_ws = max(0.0, 1 - min(ratio, 1.0) ** (1 / 3))
            else:
                p_ws = 0.3
            p_ws = max(0.05, min(0.6, p_ws))
            logger.info(
                f"Stage 1: WS graph (N={n}, k={k_ws}, p={p_ws:.3f}) "
                f"for clustering target={target_c:.3f}"
            )
            G = nx.watts_strogatz_graph(n, k_ws, p_ws)
            cur_m = G.number_of_edges()
            if cur_m != m:
                diff = m - cur_m
                nodes_list = list(G.nodes())
                if diff > 0:
                    for _ in range(diff * 5):
                        u, v = np.random.choice(nodes_list, 2, replace=False)
                        if not G.has_edge(u, v):
                            G.add_edge(u, v)
                            if G.number_of_edges() >= m:
                                break
                elif diff < 0:
                    nt = [(u, v) for u, v in G.edges()
                          if len(set(G.neighbors(u)) & set(G.neighbors(v))) == 0]
                    for u, v in nt[:abs(diff)]:
                        G.remove_edge(u, v)
                        if G.number_of_edges() <= m:
                            break

        else:
            logger.info(f"Stage 1: ER graph (N={n}, <k>={avg_deg:.2f})")
            G = nx.gnm_random_graph(n, m)

        G = nx.convert_node_labels_to_integers(G)

        if self.verbose:
            actual_m = G.number_of_edges()
            actual_avg = 2 * actual_m / n
            logger.info(f"  Generated: M={actual_m}, <k>={actual_avg:.3f}")

        return G

    def verify(self, G: nx.Graph) -> dict[str, float]:
        from ..metrics import avg_degree
        return {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "avg_degree": avg_degree(G),
        }
