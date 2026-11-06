from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh

if TYPE_CHECKING:
    from ..constraint_dag import ConstraintSpec

logger = logging.getLogger(__name__)


def _compute_fiedler_vector(G: nx.Graph) -> np.ndarray | None:
    n = G.number_of_nodes()
    if n < 3:
        return None
    try:
        L = nx.normalized_laplacian_matrix(G).astype(float)
        k = min(3, n - 1)
        vals, vecs = eigsh(L, k=k, which="SM", tol=1e-3)
        idx = np.argsort(vals)
        fiedler = np.real(vecs[:, idx[1]])
        return fiedler
    except Exception:
        return None


def _fiedler_indexed(G: nx.Graph) -> dict[int, float] | None:
    nodes = list(G.nodes())
    fvec = _compute_fiedler_vector(G)
    if fvec is None:
        return None
    return {nodes[i]: float(fvec[i]) for i in range(len(nodes))}


def _in_triangle(G: nx.Graph, u: int, v: int) -> bool:
    return len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0


def _try_increase_spectral_gap(
    G: nx.Graph,
    fiedler_map: dict[int, float],
    max_attempts: int = 200,
) -> bool:
    edges = list(G.edges())
    rng = random.Random()

    for _ in range(max_attempts):
        u, a = rng.choice(edges)
        if _in_triangle(G, u, a):
            continue
        phi_u = fiedler_map.get(u, 0.0)
        phi_a = fiedler_map.get(a, 0.0)
        if abs(phi_u - phi_a) > 0.3:
            continue

        candidates_v = [
            v for v in G.nodes()
            if v != u and v != a
            and not G.has_edge(u, v)
            and abs(fiedler_map.get(v, 0.0) - phi_u) > 0.2
        ]
        if not candidates_v:
            continue
        v = rng.choice(candidates_v)

        nbrs_v = list(G.neighbors(v))
        candidates_b = [
            b for b in nbrs_v
            if b != u and b != a
            and not G.has_edge(a, b)
            and not _in_triangle(G, v, b)
        ]
        if not candidates_b:
            continue
        b = rng.choice(candidates_b)

        if len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0:
            continue
        if len(set(G.neighbors(a)) & set(G.neighbors(b))) > 0:
            continue

        G.remove_edge(u, a)
        G.remove_edge(v, b)
        G.add_edge(u, v)
        G.add_edge(a, b)
        return True

    return False


def _try_decrease_spectral_gap(
    G: nx.Graph,
    fiedler_map: dict[int, float],
    max_attempts: int = 200,
) -> bool:
    edges = list(G.edges())
    rng = random.Random()

    for _ in range(max_attempts):
        u, v = rng.choice(edges)
        if _in_triangle(G, u, v):
            continue
        phi_u = fiedler_map.get(u, 0.0)
        phi_v = fiedler_map.get(v, 0.0)
        if abs(phi_u - phi_v) < 0.2:
            continue

        candidates_ab = []
        for _ in range(50):
            a, b = rng.choice(edges)
            if a == u or a == v or b == u or b == v:
                continue
            if _in_triangle(G, a, b):
                continue
            phi_a = fiedler_map.get(a, 0.0)
            phi_b = fiedler_map.get(b, 0.0)
            if abs(phi_a - phi_b) > 0.3:
                continue
            candidates_ab.append((a, b))
            break

        if not candidates_ab:
            continue
        a, b = candidates_ab[0]

        if not G.has_edge(u, a) and not G.has_edge(v, b):
            if len(set(G.neighbors(u)) & set(G.neighbors(a))) > 0:
                continue
            if len(set(G.neighbors(v)) & set(G.neighbors(b))) > 0:
                continue
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, a)
            G.add_edge(v, b)
            return True

    return False


class Stage5SpectralGap:

    def __init__(
        self,
        spec: "ConstraintSpec",
        max_iterations: int = 10_000,
        check_interval: int = 200,
        fiedler_recompute_interval: int = 500,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        self.max_iterations = max_iterations
        self.check_interval = check_interval
        self.fiedler_recompute_interval = fiedler_recompute_interval
        self.verbose = verbose

    def run(self, G: nx.Graph) -> nx.Graph:
        target = self.spec.spectral_gap
        if target is None:
            logger.info("Stage 5: no spectral gap target, skipping.")
            return G

        tol = self.spec.tolerances.get("spectral_gap", 0.05)
        current = self._compute_sg(G)
        logger.info(f"Stage 5: spectral gap target={target:.4f}, current={current:.4f}")

        fiedler_map = _fiedler_indexed(G) or {}
        n_success = 0

        for i in range(self.max_iterations):
            if i % self.fiedler_recompute_interval == 0:
                fiedler_map = _fiedler_indexed(G) or fiedler_map

            if i % self.check_interval == 0:
                current = self._compute_sg(G)
                if abs(current - target) <= tol:
                    logger.info(f"  Converged at iter {i}: spectral_gap={current:.4f}")
                    break

            if current < target:
                ok = _try_increase_spectral_gap(G, fiedler_map)
            else:
                ok = _try_decrease_spectral_gap(G, fiedler_map)

            if ok:
                n_success += 1

        final = self._compute_sg(G)
        logger.info(
            f"Stage 5 done: spectral_gap={final:.4f} "
            f"(target={target:.4f}, |Δ|={abs(final-target):.4f}, swaps={n_success})"
        )
        return G

    def _compute_sg(self, G: nx.Graph) -> float:
        n = G.number_of_nodes()
        if n < 3:
            return 0.0
        try:
            L = nx.normalized_laplacian_matrix(G).astype(float)
            vals = eigsh(L, k=min(3, n - 1), which="SM",
                         return_eigenvectors=False, tol=1e-3)
            vals = sorted(np.real(vals))
            return float(max(0.0, vals[1]))
        except Exception:
            return 0.0
