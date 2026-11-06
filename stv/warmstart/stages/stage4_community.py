from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from ..constraint_dag import ConstraintSpec

logger = logging.getLogger(__name__)


def _detect_communities(G: nx.Graph) -> dict[int, int]:
    try:
        import community as community_louvain
        return community_louvain.best_partition(G)
    except ImportError:
        pass
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        node_to_comm = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = cid
        return node_to_comm
    except Exception:
        return {n: 0 for n in G.nodes()}


def _compute_modularity(G: nx.Graph, partition: dict[int, int]) -> float:
    communities = {}
    for node, cid in partition.items():
        communities.setdefault(cid, set()).add(node)
    try:
        return nx.algorithms.community.quality.modularity(G, list(communities.values()))
    except Exception:
        return 0.0


def _in_triangle(G: nx.Graph, u: int, v: int) -> bool:
    return len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0


def _would_form_triangle(G: nx.Graph, u: int, v: int) -> bool:
    return len(set(G.neighbors(u)) & set(G.neighbors(v))) > 0


def _try_increase_modularity(
    G: nx.Graph,
    partition: dict[int, int],
    max_attempts: int = 300,
) -> bool:
    edges = list(G.edges())
    rng = random.Random()

    inter_edges = [(u, v) for u, v in edges
                   if partition.get(u, -1) != partition.get(v, -2)
                   and not _in_triangle(G, u, v)]
    if len(inter_edges) < 2:
        return False

    for _ in range(max_attempts):
        u, v = rng.choice(inter_edges)
        comm_u = partition.get(u, -1)
        comm_v = partition.get(v, -2)

        for _ in range(50):
            a, b_node = rng.choice(inter_edges)
            if a == u or a == v or b_node == u or b_node == v:
                continue
            if _in_triangle(G, a, b_node):
                continue

            if (partition.get(a, -1) == comm_v and partition.get(b_node, -2) == comm_u):
                if (not G.has_edge(u, b_node) and not G.has_edge(v, a)
                        and not _would_form_triangle(G, u, b_node)
                        and not _would_form_triangle(G, v, a)):
                    G.remove_edge(u, v)
                    G.remove_edge(a, b_node)
                    G.add_edge(u, b_node)
                    G.add_edge(v, a)
                    return True

            if (partition.get(a, -1) == comm_u and partition.get(b_node, -2) == comm_v):
                if (not G.has_edge(u, a) and not G.has_edge(v, b_node)
                        and not _would_form_triangle(G, u, a)
                        and not _would_form_triangle(G, v, b_node)):
                    G.remove_edge(u, v)
                    G.remove_edge(a, b_node)
                    G.add_edge(u, a)
                    G.add_edge(v, b_node)
                    return True

    return False


def _try_decrease_modularity(
    G: nx.Graph,
    partition: dict[int, int],
    max_attempts: int = 300,
) -> bool:
    edges = list(G.edges())
    rng = random.Random()
    comm_ids = list(set(partition.values()))
    if len(comm_ids) < 2:
        return False

    for _ in range(max_attempts):
        u, v = rng.choice(edges)
        if partition.get(u, -1) != partition.get(v, -2):
            continue
        if _in_triangle(G, u, v):
            continue

        for _ in range(50):
            a, b = rng.choice(edges)
            if a == u or a == v or b == u or b == v:
                continue
            if partition.get(a, -1) == partition.get(b, -2):
                continue
            if _in_triangle(G, a, b):
                continue
            if not G.has_edge(u, a) and not G.has_edge(v, b):
                if not _would_form_triangle(G, u, a) and not _would_form_triangle(G, v, b):
                    G.remove_edge(u, v)
                    G.remove_edge(a, b)
                    G.add_edge(u, a)
                    G.add_edge(v, b)
                    return True
            break

    return False


class Stage4Community:

    def __init__(
        self,
        spec: "ConstraintSpec",
        max_iterations: int | None = None,
        check_interval: int = 300,
        redetect_interval: int = 1000,
        verbose: bool = False,
    ) -> None:
        self.spec = spec
        n = spec.n_nodes
        k = int(2 * (spec.n_edges or 1) / max(n, 1))
        density_factor = max(0.1, min(1.0, 6.0 / max(k, 6)))
        self.max_iterations = max_iterations if max_iterations is not None else max(
            20_000, int(n * k * 20 * density_factor)
        )
        self.check_interval = check_interval
        self.redetect_interval = max(2000, int(k * 300))
        self.verbose = verbose

    def run(self, G: nx.Graph) -> nx.Graph:
        target = self.spec.modularity
        if target is None:
            logger.info("Stage 4: no modularity target, skipping.")
            return G

        tol = self.spec.tolerances.get("modularity", 0.05)
        partition = _detect_communities(G)
        current = _compute_modularity(G, partition)
        logger.info(
            f"Stage 4: modularity target={target:.4f}, current={current:.4f}, "
            f"communities={len(set(partition.values()))}"
        )

        n_success = 0
        last_success_iter = 0

        for i in range(self.max_iterations):
            if i % self.redetect_interval == 0:
                partition = _detect_communities(G)

            if i % self.check_interval == 0:
                fresh_partition = _detect_communities(G)
                current = _compute_modularity(G, fresh_partition)
                partition = fresh_partition
                if abs(current - target) <= tol:
                    logger.info(f"  Converged at iter {i}: modularity={current:.4f}")
                    break

            if i - last_success_iter > self.check_interval * 3:
                logger.warning(
                    f"Stage 4: no successful swaps for {i - last_success_iter} iters. "
                    f"Stopping early at iter {i}."
                )
                break

            if current < target:
                ok = _try_increase_modularity(G, partition)
            else:
                ok = _try_decrease_modularity(G, partition)

            if ok:
                n_success += 1
                last_success_iter = i

        final_partition = _detect_communities(G)
        final_q = _compute_modularity(G, final_partition)
        logger.info(
            f"Stage 4 done: modularity={final_q:.4f} "
            f"(target={target:.4f}, |Δ|={abs(final_q-target):.4f}, swaps={n_success})"
        )
        return G
