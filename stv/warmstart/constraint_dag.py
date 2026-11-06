from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STAGE_ORDER: list[str] = [
    "degree_dist",
    "clustering",
    "avg_path_length",
    "modularity",
    "spectral_gap",
    "assortativity",
]

METRIC_ALIASES: dict[str, str] = {
    "degree_distribution": "degree_dist",
    "cc": "clustering",
    "clustering_coefficient": "clustering",
    "apl": "avg_path_length",
    "path_length": "avg_path_length",
    "community": "modularity",
    "q": "modularity",
    "lambda2": "spectral_gap",
    "fiedler": "spectral_gap",
    "algebraic_connectivity": "spectral_gap",
}

METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "clustering": (0.0, 1.0),
    "avg_path_length": (1.0, float("inf")),
    "assortativity": (-1.0, 1.0),
    "modularity": (-0.5, 1.0),
    "spectral_gap": (0.0, 2.0),
    "global_efficiency": (0.0, 1.0),
    "transitivity": (0.0, 1.0),
    "lcc_fraction": (0.0, 1.0),
}

CONFLICT_HINTS: list[tuple[str, str, str]] = [
    ("clustering", "avg_path_length",
     "High clustering and very short path length are compatible only at high density."),
    ("modularity", "avg_path_length",
     "High modularity tends to increase average path length."),
    ("spectral_gap", "modularity",
     "High spectral gap conflicts with strong community structure."),
]


@dataclass
class ConstraintSpec:

    n_nodes: int
    n_edges: int | None = None
    avg_degree: float | None = None
    degree_gamma: float | None = None
    degree_seq: list[int] | None = None

    clustering: float | None = None
    avg_path_length: float | None = None
    assortativity: float | None = None
    modularity: float | None = None
    spectral_gap: float | None = None

    tolerances: dict[str, float] = field(default_factory=lambda: {
        "clustering": 0.05,
        "avg_path_length": 0.3,
        "assortativity": 0.05,
        "modularity": 0.05,
        "spectral_gap": 0.05,
        "degree_dist": 0.1,
    })
    weights: dict[str, float] = field(default_factory=lambda: {
        "degree_dist": 2.0,
        "clustering": 1.5,
        "avg_path_length": 1.0,
        "modularity": 1.0,
        "spectral_gap": 0.5,
    })

    def __post_init__(self) -> None:
        if self.n_edges is None and self.avg_degree is not None:
            self.n_edges = int(round(self.avg_degree * self.n_nodes / 2))
        if self.n_edges is None and self.avg_degree is None:
            raise ValueError("Provide either n_edges or avg_degree.")

    @property
    def scalar_targets(self) -> dict[str, float]:
        candidates = {
            "clustering": self.clustering,
            "avg_path_length": self.avg_path_length,
            "assortativity": self.assortativity,
            "modularity": self.modularity,
            "spectral_gap": self.spectral_gap,
        }
        return {k: v for k, v in candidates.items() if v is not None}

    @property
    def active_stages(self) -> list[str]:
        targets = set(self.scalar_targets.keys())
        return [s for s in STAGE_ORDER if s in targets or s == "degree_dist"]

    def resolve_alias(self, name: str) -> str:
        return METRIC_ALIASES.get(name, name)

    def check_bounds(self) -> list[str]:
        warnings_list = []
        for name, target in self.scalar_targets.items():
            lo, hi = METRIC_BOUNDS.get(name, (-float("inf"), float("inf")))
            if not (lo <= target <= hi):
                warnings_list.append(
                    f"Target {name}={target:.3f} is outside valid range [{lo}, {hi}]."
                )
        return warnings_list

    def detect_conflicts(self) -> list[str]:
        hints = []
        targets = self.scalar_targets

        if "clustering" in targets and "avg_path_length" in targets:
            density = 2 * self.n_edges / (self.n_nodes * (self.n_nodes - 1))
            if targets["clustering"] > 0.5 and targets["avg_path_length"] < 3.0 and density < 0.1:
                hints.append(
                    "WARNING: High clustering (>0.5) and short path length (<3) at low density "
                    "(<0.1) may conflict. Consider increasing avg_degree."
                )

        if "spectral_gap" in targets and "modularity" in targets:
            if targets["spectral_gap"] > 0.3 and targets["modularity"] > 0.6:
                hints.append(
                    "WARNING: High spectral gap (>0.3) and high modularity (>0.6) conflict — "
                    "strong community structure implies small spectral gap."
                )

        return hints

    def summary(self) -> str:
        lines = [
            f"ConstraintSpec: N={self.n_nodes}, M={self.n_edges}",
            f"  degree_gamma = {self.degree_gamma}",
        ]
        for k, v in self.scalar_targets.items():
            tol = self.tolerances.get(k, 0.05)
            lines.append(f"  {k} = {v:.4f}  (±{tol})")
        return "\n".join(lines)


def inheritance_check(
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    targets: dict[str, float],
    tolerances: dict[str, float],
    stage_idx: int,
) -> dict[str, bool]:
    preserved = {}
    for i, metric in enumerate(STAGE_ORDER[:stage_idx]):
        if metric not in targets:
            continue
        target = targets[metric]
        tol = tolerances.get(metric, 0.05)
        val_after = metrics_after.get(metric)
        if val_after is None:
            preserved[metric] = False
        else:
            preserved[metric] = abs(float(val_after) - float(target)) <= tol * 1.5
    return preserved
