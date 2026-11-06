from __future__ import annotations

import warnings
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh


def degree_distribution(G: nx.Graph) -> np.ndarray:
    if G.number_of_nodes() == 0:
        return np.array([0.0])
    degrees = [d for _, d in G.degree()]
    max_deg = max(degrees) if degrees else 0
    hist = np.zeros(max_deg + 1)
    for d in degrees:
        hist[d] += 1
    hist /= G.number_of_nodes()
    return hist


def avg_degree(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    return 2 * G.number_of_edges() / n if n > 0 else 0.0


def clustering_coefficient(G: nx.Graph) -> float:
    return nx.average_clustering(G)


def avg_path_length(G: nx.Graph) -> float:
    if G.number_of_nodes() <= 1:
        return 0.0
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes)
    if G_lcc.number_of_nodes() <= 1:
        return 0.0
    try:
        return nx.average_shortest_path_length(G_lcc)
    except nx.NetworkXError:
        return float("inf")


def assortativity(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    try:
        r = nx.degree_assortativity_coefficient(G)
    except Exception:
        return 0.0
    return float(r) if np.isfinite(r) else 0.0


def modularity(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        return nx.algorithms.community.quality.modularity(G, communities)
    except Exception:
        return 0.0


def spectral_gap(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    if n <= 2 or G.number_of_edges() == 0:
        return 0.0
    try:
        L = nx.normalized_laplacian_matrix(G).astype(float)
        vals = eigsh(L, k=min(3, n - 1), which="SM", return_eigenvectors=False, tol=1e-4)
        vals = sorted(np.real(vals))
        return float(max(0.0, vals[1]))
    except Exception:
        return 0.0


def global_efficiency(G: nx.Graph) -> float:
    return nx.global_efficiency(G)


def transitivity(G: nx.Graph) -> float:
    return nx.transitivity(G)


def lcc_fraction(G: nx.Graph) -> float:
    if G.number_of_nodes() == 0:
        return 0.0
    lcc = max(nx.connected_components(G), key=len)
    return len(lcc) / G.number_of_nodes()


METRIC_FUNCTIONS = {
    "clustering": clustering_coefficient,
    "avg_path_length": avg_path_length,
    "assortativity": assortativity,
    "modularity": modularity,
    "spectral_gap": spectral_gap,
    "global_efficiency": global_efficiency,
    "transitivity": transitivity,
    "lcc_fraction": lcc_fraction,
}

METRIC_SCALES = {
    "clustering": 1.0,
    "avg_path_length": 10.0,
    "assortativity": 1.0,
    "modularity": 1.0,
    "spectral_gap": 1.0,
    "global_efficiency": 1.0,
    "transitivity": 1.0,
    "lcc_fraction": 1.0,
}


def compute_metrics(G: nx.Graph, which: list[str] | None = None) -> dict[str, Any]:
    if which is None:
        which = list(METRIC_FUNCTIONS.keys())

    result: dict[str, Any] = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "avg_degree": avg_degree(G),
        "degree_dist": degree_distribution(G),
    }

    for name in which:
        fn = METRIC_FUNCTIONS.get(name)
        if fn is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result[name] = fn(G)

    return result


def constraint_violation(
    metrics: dict[str, Any],
    targets: dict[str, float],
    tolerances: dict[str, float],
) -> dict[str, float]:
    violations: dict[str, float] = {}
    for key, target in targets.items():
        if key == "degree_dist":
            emp = metrics.get("degree_dist", np.array([0.0]))
            tgt = np.asarray(target)
            max_len = max(len(emp), len(tgt))
            emp_pad = np.pad(emp, (0, max_len - len(emp)))
            tgt_pad = np.pad(tgt, (0, max_len - len(tgt)))
            diff = float(np.sum(np.abs(emp_pad - tgt_pad)))
            tol = tolerances.get(key, 0.1)
            violations[key] = max(0.0, diff - tol)
        else:
            val = metrics.get(key)
            if val is None or np.isnan(float(val)) or np.isinf(float(val)):
                violations[key] = float(abs(target)) + 1.0
            else:
                tol = tolerances.get(key, 0.05)
                violations[key] = max(0.0, abs(float(val) - float(target)) - tol)
    return violations


def total_violation(violations: dict[str, float], weights: dict[str, float] | None = None) -> float:
    if weights is None:
        return sum(violations.values())
    return sum(weights.get(k, 1.0) * v for k, v in violations.items())


def triangle_count(G: nx.Graph) -> int:
    return int(sum(nx.triangles(G).values()) // 3)


def wedge_count(G: nx.Graph) -> int:
    return sum(d * (d - 1) // 2 for _, d in G.degree())


METRIC_FUNCTIONS["triangle_count"] = triangle_count
METRIC_FUNCTIONS["wedge_count"] = wedge_count


def mean_betweenness_centrality(G: nx.Graph, normalized: bool = True, k_approx: int | None = None) -> float:
    if G.number_of_nodes() <= 1:
        return 0.0
    try:
        if k_approx is not None:
            bc = nx.betweenness_centrality(G, k=k_approx, normalized=normalized)
        else:
            bc = nx.betweenness_centrality(G, normalized=normalized)
        return float(np.mean(list(bc.values())))
    except Exception:
        return 0.0


def degree_heterogeneity(G: nx.Graph) -> float:
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    if degrees.sum() == 0:
        return 0.0
    return float(np.mean(degrees ** 2) / np.mean(degrees))


def k_core_number(G: nx.Graph) -> int:
    if G.number_of_edges() == 0:
        return 0
    core = nx.core_number(G)
    return int(max(core.values())) if core else 0


def rich_club_coefficient_mean(G: nx.Graph) -> float:
    if G.number_of_edges() == 0:
        return 0.0
    try:
        rc = nx.rich_club_coefficient(G, normalized=False)
        if not rc:
            return 0.0
        return float(np.mean(list(rc.values())))
    except Exception:
        return 0.0


def local_efficiency(G: nx.Graph) -> float:
    return nx.local_efficiency(G)


def network_diameter(G: nx.Graph) -> float:
    if G.number_of_nodes() <= 1:
        return 0.0
    lcc_nodes = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(lcc_nodes)
    if G_lcc.number_of_nodes() <= 2:
        return float(G_lcc.number_of_nodes() - 1)
    try:
        return float(nx.diameter(G_lcc))
    except Exception:
        return float("inf")


METRIC_FUNCTIONS["mean_betweenness"] = mean_betweenness_centrality
METRIC_FUNCTIONS["degree_heterogeneity"] = degree_heterogeneity
METRIC_FUNCTIONS["k_core_number"] = k_core_number
METRIC_FUNCTIONS["rich_club_mean"] = rich_club_coefficient_mean
METRIC_FUNCTIONS["local_efficiency"] = local_efficiency
METRIC_FUNCTIONS["diameter"] = network_diameter
