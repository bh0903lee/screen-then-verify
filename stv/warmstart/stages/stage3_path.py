from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ..constraint_dag import ConstraintSpec

logger = logging.getLogger(__name__)


def _non_triangle_edges(G: nx.Graph) -> list[tuple[int, int]]:
    result = []
    for u, v in G.edges():
        common = set(G.neighbors(u)) & set(G.neighbors(v))
        if len(common) == 0:
            result.append((u, v))
    return result


def _would_form_triangle(G: nx.Graph, u: int, v: int) -> bool:
    return len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0


def _try_decrease_path_length(G: nx.Graph, max_attempts: int = 10, min_dist: int = 3) -> bool:
    nt_edges = _non_triangle_edges(G)
    if len(nt_edges) < 2:
        return False

    rng = random.Random()
    lcc = max(nx.connected_components(G), key=len)
    lcc_list = list(lcc)
    if len(lcc_list) < 4:
        return False

    for _ in range(max_attempts):
        u = rng.choice(lcc_list)
        lengths = dict(nx.single_source_shortest_path_length(G, u, cutoff=6))
        far_nodes = [v for v, d in lengths.items()
                     if d >= min_dist and not G.has_edge(u, v) and v != u]
        if not far_nodes:
            continue
        v = rng.choice(far_nodes)

        nt_u = [(x, y) for (x, y) in nt_edges if x == u or y == u]
        if not nt_u:
            continue
        e_ua = rng.choice(nt_u)
        a = e_ua[1] if e_ua[0] == u else e_ua[0]
        if a == v:
            continue

        nt_v = [(x, y) for (x, y) in nt_edges if x == v or y == v]
        candidates_b = [
            e[1] if e[0] == v else e[0]
            for e in nt_v
            if (e[1] if e[0] == v else e[0]) not in (u, a)
            and not G.has_edge(a, e[1] if e[0] == v else e[0])
        ]
        if not candidates_b:
            continue
        b = rng.choice(candidates_b)

        if _would_form_triangle(G, u, v) or _would_form_triangle(G, a, b):
            continue

        G.remove_edge(u, a)
        G.remove_edge(v, b)
        G.add_edge(u, v)
        G.add_edge(a, b)
        return True

    return False


def _try_increase_path_length(G: nx.Graph, max_attempts: int = 50) -> bool:
    nt_edges = _non_triangle_edges(G)
    if len(nt_edges) < 2:
        return False

    rng = random.Random()

    for _ in range(max_attempts):
        u, v = rng.choice(nt_edges)
        candidates = [(a, b) for (a, b) in nt_edges
                      if a != u and a != v and b != u and b != v]
        if not candidates:
            continue
        a, b = rng.choice(candidates)

        if not G.has_edge(u, a) and not G.has_edge(v, b):
            if _would_form_triangle(G, u, a) or _would_form_triangle(G, v, b):
                continue
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, a)
            G.add_edge(v, b)
            return True
        elif not G.has_edge(u, b) and not G.has_edge(v, a):
            if _would_form_triangle(G, u, b) or _would_form_triangle(G, v, a):
                continue
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, b)
            G.add_edge(v, a)
            return True

    return False


class Stage3PathLength:

    def __init__(
        self,
        spec: "ConstraintSpec",
        max_iterations: int | None = None,
        check_interval: int = 200,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        n = spec.n_nodes
        k = int(2 * (spec.n_edges or 1) / max(n, 1))
        self.max_iterations = max_iterations if max_iterations is not None else max(5_000, n * k * 3)
        self.check_interval = check_interval
        self.verbose = verbose

    def run(self, G: nx.Graph) -> nx.Graph:
        target = self.spec.avg_path_length
        if target is None:
            logger.info("Stage 3: no path length target, skipping.")
            return G

        tol = self.spec.tolerances.get("avg_path_length", 0.3)
        current = self._compute_apl(G)
        initial_current = current
        logger.info(f"Stage 3: path length target={target:.4f}, current={current:.4f}")

        min_shortcut_dist = max(2, int(round(target)) - 1)

        n_success = 0

        for i in range(self.max_iterations):
            if i % self.check_interval == 0:
                current = self._compute_apl(G)
                if abs(current - target) <= tol:
                    logger.info(f"  Converged at iter {i}: apl={current:.4f}")
                    break

            if i == int(self.max_iterations * 0.2):
                check_current = self._compute_apl(G)
                if abs(check_current - initial_current) < 0.02:
                    logger.warning(
                        f"Stage 3: APL not moving after {i} iters "
                        f"(initial={initial_current:.3f}, now={check_current:.3f}). "
                        f"Target may be infeasible. Stopping early."
                    )
                    break

            if current > target:
                ok = _try_decrease_path_length(G, min_dist=min_shortcut_dist)
            else:
                ok = _try_increase_path_length(G)

            if ok:
                n_success += 1

        final = self._compute_apl(G)
        logger.info(
            f"Stage 3 done: apl={final:.4f} "
            f"(target={target:.4f}, |Δ|={abs(final-target):.4f}, swaps={n_success})"
        )
        return G

    def _compute_apl(self, G: nx.Graph) -> float:
        lcc = max(nx.connected_components(G), key=len)
        G_lcc = G.subgraph(lcc)
        if G_lcc.number_of_nodes() <= 1:
            return float("inf")
        try:
            return nx.average_shortest_path_length(G_lcc)
        except Exception:
            return float("inf")
