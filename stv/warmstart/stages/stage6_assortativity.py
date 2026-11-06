from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ..constraint_dag import ConstraintSpec

logger = logging.getLogger(__name__)


def _delta_s(du: int, dv: int, da: int, db: int) -> int:
    return (da - dv) * (du - db)


def _is_triangle_edge(G: nx.Graph, u: int, v: int) -> bool:
    return len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0


def _try_increase_assortativity(
    G: nx.Graph,
    degrees: dict,
    max_attempts: int = 300,
    preserve_clustering: bool = True,
) -> bool:
    edges = list(G.edges())
    if len(edges) < 2:
        return False
    rng = random.Random()

    for _ in range(max_attempts):
        e1, e2 = rng.sample(edges, 2)
        u, v = e1
        a, b = e2

        if len({u, v, a, b}) < 4:
            continue

        if preserve_clustering:
            if _is_triangle_edge(G, u, v) or _is_triangle_edge(G, a, b):
                continue

        du, dv, da, db = degrees[u], degrees[v], degrees[a], degrees[b]

        ds1 = _delta_s(du, dv, da, db)
        if ds1 > 0 and not G.has_edge(u, a) and not G.has_edge(v, b) and u != b and v != a:
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, a)
            G.add_edge(v, b)
            return True

        ds2 = _delta_s(du, dv, db, da)
        if ds2 > 0 and not G.has_edge(u, b) and not G.has_edge(v, a) and u != a and v != b:
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, b)
            G.add_edge(v, a)
            return True

    return False


def _try_decrease_assortativity(
    G: nx.Graph,
    degrees: dict,
    max_attempts: int = 300,
    preserve_clustering: bool = True,
) -> bool:
    edges = list(G.edges())
    if len(edges) < 2:
        return False
    rng = random.Random()

    for _ in range(max_attempts):
        e1, e2 = rng.sample(edges, 2)
        u, v = e1
        a, b = e2

        if len({u, v, a, b}) < 4:
            continue

        if preserve_clustering:
            if _is_triangle_edge(G, u, v) or _is_triangle_edge(G, a, b):
                continue

        du, dv, da, db = degrees[u], degrees[v], degrees[a], degrees[b]

        ds1 = _delta_s(du, dv, da, db)
        if ds1 < 0 and not G.has_edge(u, a) and not G.has_edge(v, b) and u != b and v != a:
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, a)
            G.add_edge(v, b)
            return True

        ds2 = _delta_s(du, dv, db, da)
        if ds2 < 0 and not G.has_edge(u, b) and not G.has_edge(v, a) and u != a and v != b:
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, b)
            G.add_edge(v, a)
            return True

    return False


class Stage6Assortativity:

    def __init__(
        self,
        spec: "ConstraintSpec",
        max_iterations: int | None = None,
        check_interval: int = 100,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        n = spec.n_nodes
        k = max(1, int(2 * (spec.n_edges or 1) / max(n, 1)))
        self.max_iterations = max_iterations if max_iterations is not None else max(20_000, n * k * 10)
        self.check_interval = check_interval
        self.verbose = verbose

    def run(self, G: nx.Graph) -> nx.Graph:
        target = self.spec.assortativity
        if target is None:
            logger.info("Stage 6: no assortativity target, skipping.")
            return G

        tol = self.spec.tolerances.get("assortativity", 0.05)
        current = nx.degree_assortativity_coefficient(G)
        logger.info(f"Stage 6: assortativity target={target:.4f}, current={current:.4f}")

        if abs(current - target) <= tol:
            logger.info(f"Stage 6: already satisfied (r={current:.4f}).")
            return G

        degrees = dict(G.degree())

        n_success = 0
        n_fail = 0

        for i in range(self.max_iterations):
            if i % self.check_interval == 0:
                current = nx.degree_assortativity_coefficient(G)
                if abs(current - target) <= tol:
                    logger.info(f"  Converged at iter {i}: r={current:.4f}")
                    break

            if i > self.max_iterations * 0.3 and n_success == 0:
                logger.warning(f"Stage 6: no successful swaps in {i} iters. Stopping.")
                break

            if current < target:
                ok = _try_increase_assortativity(G, degrees)
            else:
                ok = _try_decrease_assortativity(G, degrees)

            if ok:
                n_success += 1
            else:
                n_fail += 1

        final = nx.degree_assortativity_coefficient(G)
        logger.info(
            f"Stage 6 done: r={final:.4f} "
            f"(target={target:.4f}, |Δ|={abs(final-target):.4f}, "
            f"swaps={n_success}, failures={n_fail})"
        )
        return G
