from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from ..constraint_dag import ConstraintSpec

logger = logging.getLogger(__name__)


def _would_form_triangle(G: nx.Graph, u: int, v: int) -> bool:
    return len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0


def _estimate_max_clustering(G: nx.Graph) -> float:
    degrees = [d for _, d in G.degree()]
    if not degrees:
        return 0.0
    avg_k = float(np.mean(degrees))
    if avg_k <= 2:
        return 0.0
    return 3.0 * (avg_k - 2) / (4.0 * (avg_k - 1))


def _try_increase_clustering(G: nx.Graph, max_attempts: int = 200) -> bool:
    nodes = list(G.nodes())
    rng = random.Random()

    for _ in range(max_attempts):
        u = rng.choice(nodes)
        nbrs_u = list(G.neighbors(u))
        if len(nbrs_u) < 1:
            continue

        non_nbrs_u = [n for n in nodes if n != u and not G.has_edge(u, n)]
        if not non_nbrs_u:
            continue

        nbrs_u_set = set(nbrs_u)
        candidates_v = [v for v in non_nbrs_u
                        if len(nbrs_u_set & set(G.neighbors(v))) > 0]
        if not candidates_v:
            continue

        v = rng.choice(candidates_v)
        nbrs_v = list(G.neighbors(v))

        common_uv = nbrs_u_set & set(G.neighbors(v))
        candidates_a = [
            a for a in nbrs_u
            if a != v and a not in common_uv
            and len(nbrs_u_set & set(G.neighbors(a)) - {v}) == 0
        ]
        if not candidates_a:
            continue
        a = rng.choice(candidates_a)

        nbrs_v_set = set(nbrs_v)
        candidates_b = [b for b in nbrs_v
                        if b != u and b != a and not G.has_edge(a, b)
                        and len(nbrs_v_set & set(G.neighbors(b)) - {u}) == 0]
        if not candidates_b:
            continue
        b = rng.choice(candidates_b)

        G.remove_edge(u, a)
        G.remove_edge(v, b)
        G.add_edge(u, v)
        G.add_edge(a, b)
        return True

    return False


def _try_increase_clustering_aggressive(G: nx.Graph, max_attempts: int = 500) -> bool:
    rng = random.Random()
    nodes = list(G.nodes())
    local_c = nx.clustering(G)
    avg_c = float(np.mean(list(local_c.values()))) if local_c else 0.0
    target_nodes = [n for n in nodes if local_c.get(n, 0) < avg_c + 0.05]
    if not target_nodes:
        target_nodes = nodes

    for _ in range(max_attempts):
        u = rng.choice(target_nodes)
        nbrs_u = list(G.neighbors(u))
        if len(nbrs_u) < 1:
            continue
        rng.shuffle(nbrs_u)
        found = False
        for a in nbrs_u[:5]:
            nbrs_a = [x for x in G.neighbors(a) if x != u and not G.has_edge(u, x)]
            if not nbrs_a:
                continue
            b = rng.choice(nbrs_a)
            nt_b = [(b, x) for x in G.neighbors(b)
                    if x != a and x != u
                    and len(set(G.neighbors(b)) & set(G.neighbors(x))) == 0]
            if not nt_b:
                continue
            b, x = rng.choice(nt_b)
            nt_u = [(u, y) for y in G.neighbors(u)
                    if y != a and y != b
                    and len(set(G.neighbors(u)) & set(G.neighbors(y))) == 0]
            if not nt_u:
                continue
            u2, y = rng.choice(nt_u)
            if G.has_edge(x, y) or x == y:
                continue
            if _would_form_triangle(G, x, y):
                pass
            G.remove_edge(u, y)
            G.remove_edge(b, x)
            G.add_edge(u, b)
            G.add_edge(x, y)
            found = True
            break
        if found:
            return True
    return False


def _try_decrease_clustering(G: nx.Graph, max_attempts: int = 200) -> bool:
    edges = list(G.edges())
    if not edges:
        return False
    rng = random.Random()
    nodes = list(G.nodes())

    for _ in range(max_attempts):
        u, v = rng.choice(edges)
        common_uv = set(G.neighbors(u)) & set(G.neighbors(v))
        if not common_uv:
            continue

        candidates_ab = []
        for _ in range(50):
            a, b = rng.choice(edges)
            if a == u or a == v or b == u or b == v:
                continue
            if (set(G.neighbors(a)) & set(G.neighbors(b))):
                continue
            candidates_ab.append((a, b))
            if candidates_ab:
                break

        if not candidates_ab:
            continue
        a, b = candidates_ab[0]

        if G.has_edge(u, a) or G.has_edge(v, b):
            if G.has_edge(u, b) or G.has_edge(v, a):
                continue
            G.remove_edge(u, v)
            G.remove_edge(a, b)
            G.add_edge(u, b)
            G.add_edge(v, a)
            return True

        G.remove_edge(u, v)
        G.remove_edge(a, b)
        G.add_edge(u, a)
        G.add_edge(v, b)
        return True

    return False


class Stage2Clustering:

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
        self.max_iterations = max_iterations if max_iterations is not None else max(100_000, n * k * 50)
        self.check_interval = check_interval
        self.verbose = verbose

    def run(self, G: nx.Graph) -> nx.Graph:
        target = self.spec.clustering
        if target is None:
            logger.info("Stage 2: no clustering target, skipping.")
            return G

        tol = self.spec.tolerances.get("clustering", 0.05)
        current = nx.average_clustering(G)
        logger.info(f"Stage 2: clustering target={target:.4f}, current={current:.4f}")

        c_max = _estimate_max_clustering(G)
        effective_target = min(target, c_max + 0.05)
        if target > c_max + 0.1:
            logger.warning(
                f"Stage 2: target clustering={target:.3f} may exceed WS upper bound "
                f"c_max≈{c_max:.3f}. Using effective_target={effective_target:.3f}."
            )

        n_success = 0
        n_fail = 0
        stuck_counter = 0
        started_above_target = (current > effective_target)

        for i in range(self.max_iterations):
            in_decrease_mode = (current > effective_target)
            update_interval = min(self.check_interval, 10) if in_decrease_mode else self.check_interval

            if i % update_interval == 0:
                current = nx.average_clustering(G)
                if abs(current - effective_target) <= tol:
                    logger.info(f"  Converged at iter {i}: clustering={current:.4f}")
                    break
                if started_above_target and current < effective_target - tol:
                    logger.warning(
                        f"Stage 2: overshot decrease target at iter {i}: "
                        f"current={current:.4f}, target={effective_target:.4f}. Stopping."
                    )
                    break

            if i > self.max_iterations * 0.3 and n_success == 0:
                logger.warning(
                    f"Stage 2: no successful swaps after {i} iters. Stopping early."
                )
                break

            if current < effective_target:
                if stuck_counter > 2000:
                    ok = _try_increase_clustering_aggressive(G)
                else:
                    ok = _try_increase_clustering(G)
            else:
                ok = _try_decrease_clustering(G)

            if ok:
                n_success += 1
                stuck_counter = 0
            else:
                n_fail += 1
                stuck_counter += 1

        final = nx.average_clustering(G)
        logger.info(
            f"Stage 2 done: clustering={final:.4f} "
            f"(target={target:.4f}, |Δ|={abs(final-target):.4f}, "
            f"swaps={n_success}, failures={n_fail})"
        )
        return G

    def verify(self, G_before: nx.Graph, G_after: nx.Graph) -> dict[str, float]:
        deg_before = sorted(d for _, d in G_before.degree())
        deg_after = sorted(d for _, d in G_after.degree())
        degree_preserved = deg_before == deg_after
        return {
            "degree_preserved": float(degree_preserved),
            "clustering_before": nx.average_clustering(G_before),
            "clustering_after": nx.average_clustering(G_after),
        }
