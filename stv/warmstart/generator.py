from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .constraint_dag import ConstraintSpec, inheritance_check
from .metrics import compute_metrics, constraint_violation
from .stages import (
    Stage1DegreeSequence,
    Stage2Clustering,
    Stage3PathLength,
    Stage4Community,
    Stage5SpectralGap,
    Stage6Assortativity,
)

logger = logging.getLogger(__name__)


@dataclass
class StageReport:
    stage_name: str
    metrics_before: dict[str, Any]
    metrics_after: dict[str, Any]
    violations_after: dict[str, float]
    inheritance_check: dict[str, bool]
    elapsed_sec: float


@dataclass
class GenerationReport:
    spec: ConstraintSpec
    stage_reports: list[StageReport] = field(default_factory=list)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    final_violations: dict[str, float] = field(default_factory=dict)
    satisfied: bool = False
    total_elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = ["=" * 60, "Warm-start generation report", "=" * 60]
        lines.append(self.spec.summary())
        lines.append("")
        lines.append("Per-stage results:")
        for r in self.stage_reports:
            violations_str = ", ".join(
                f"{k}={v:.4f}" for k, v in r.violations_after.items()
            )
            lines.append(
                f"  {r.stage_name:30s} | violations: [{violations_str}]"
                f" | {r.elapsed_sec:.1f}s"
            )
        lines.append("")
        lines.append("Final metrics:")
        for k, v in self.final_metrics.items():
            if k not in ("degree_dist", "n_nodes", "n_edges"):
                lines.append(f"  {k:30s} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")
        lines.append(f"\nAll constraints satisfied: {self.satisfied}")
        lines.append(f"Total time: {self.total_elapsed_sec:.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


class WarmStartGenerator:

    def __init__(
        self,
        spec: ConstraintSpec,
        stage_configs: dict[str, dict] | None = None,
        verbose: bool = False,
        n_passes: int = 1,
    ) -> None:
        self.spec = spec
        self.stage_configs = stage_configs or {}
        self.verbose = verbose
        self.n_passes = max(1, n_passes)
        self.last_report: GenerationReport | None = None

        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
                datefmt="%H:%M:%S",
            )

        bound_warnings = spec.check_bounds()
        for w in bound_warnings:
            logger.warning(w)

        conflict_hints = spec.detect_conflicts()
        for h in conflict_hints:
            logger.warning(h)

    def generate(self) -> nx.Graph:
        t0 = time.time()
        report = GenerationReport(spec=self.spec)
        targets = self.spec.scalar_targets
        tolerances = self.spec.tolerances

        t1 = time.time()
        stage1 = Stage1DegreeSequence(
            self.spec,
            **self.stage_configs.get("stage1", {}),
            verbose=self.verbose,
        )
        G = stage1.run()
        metrics_s1 = compute_metrics(G, list(targets.keys()))
        violations_s1 = constraint_violation(metrics_s1, targets, tolerances)
        report.stage_reports.append(StageReport(
            stage_name="Stage 1: Degree Sequence",
            metrics_before={},
            metrics_after=metrics_s1,
            violations_after={k: violations_s1.get(k, 0.0) for k in ["degree_dist"]},
            inheritance_check={},
            elapsed_sec=time.time() - t1,
        ))

        if self.spec.clustering is not None:
            t2 = time.time()
            G_before = G.copy()
            metrics_before = compute_metrics(G_before, list(targets.keys()))
            stage2 = Stage2Clustering(
                self.spec,
                **self.stage_configs.get("stage2", {}),
                verbose=self.verbose,
            )
            G = stage2.run(G)
            metrics_after = compute_metrics(G, list(targets.keys()))
            violations_after = constraint_violation(metrics_after, targets, tolerances)
            inh = inheritance_check(metrics_before, metrics_after, targets, tolerances, 2)
            report.stage_reports.append(StageReport(
                stage_name="Stage 2: Clustering",
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                violations_after={k: violations_after.get(k, 0.0)
                                   for k in ["clustering"]},
                inheritance_check=inh,
                elapsed_sec=time.time() - t2,
            ))

        if self.spec.avg_path_length is not None:
            t3 = time.time()
            metrics_before = compute_metrics(G, list(targets.keys()))
            stage3 = Stage3PathLength(
                self.spec,
                **self.stage_configs.get("stage3", {}),
                verbose=self.verbose,
            )
            G = stage3.run(G)
            metrics_after = compute_metrics(G, list(targets.keys()))
            violations_after = constraint_violation(metrics_after, targets, tolerances)
            inh = inheritance_check(metrics_before, metrics_after, targets, tolerances, 3)
            report.stage_reports.append(StageReport(
                stage_name="Stage 3: Path Length",
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                violations_after={k: violations_after.get(k, 0.0)
                                   for k in ["avg_path_length"]},
                inheritance_check=inh,
                elapsed_sec=time.time() - t3,
            ))

        if self.spec.modularity is not None:
            t4 = time.time()
            metrics_before = compute_metrics(G, list(targets.keys()))
            stage4 = Stage4Community(
                self.spec,
                **self.stage_configs.get("stage4", {}),
                verbose=self.verbose,
            )
            G = stage4.run(G)
            metrics_after = compute_metrics(G, list(targets.keys()))
            violations_after = constraint_violation(metrics_after, targets, tolerances)
            inh = inheritance_check(metrics_before, metrics_after, targets, tolerances, 4)
            report.stage_reports.append(StageReport(
                stage_name="Stage 4: Modularity",
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                violations_after={k: violations_after.get(k, 0.0)
                                   for k in ["modularity"]},
                inheritance_check=inh,
                elapsed_sec=time.time() - t4,
            ))

        if self.spec.spectral_gap is not None:
            t5 = time.time()
            metrics_before = compute_metrics(G, list(targets.keys()))
            stage5 = Stage5SpectralGap(
                self.spec,
                **self.stage_configs.get("stage5", {}),
                verbose=self.verbose,
            )
            G = stage5.run(G)
            metrics_after = compute_metrics(G, list(targets.keys()))
            violations_after = constraint_violation(metrics_after, targets, tolerances)
            inh = inheritance_check(metrics_before, metrics_after, targets, tolerances, 5)
            report.stage_reports.append(StageReport(
                stage_name="Stage 5: Spectral Gap",
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                violations_after={k: violations_after.get(k, 0.0)
                                   for k in ["spectral_gap"]},
                inheritance_check=inh,
                elapsed_sec=time.time() - t5,
            ))

        if self.spec.assortativity is not None:
            t6 = time.time()
            metrics_before = compute_metrics(G, list(targets.keys()))
            stage6 = Stage6Assortativity(
                self.spec,
                **self.stage_configs.get("stage6", {}),
                verbose=self.verbose,
            )
            G = stage6.run(G)
            metrics_after = compute_metrics(G, list(targets.keys()))
            violations_after = constraint_violation(metrics_after, targets, tolerances)
            inh = inheritance_check(metrics_before, metrics_after, targets, tolerances, 6)
            report.stage_reports.append(StageReport(
                stage_name="Stage 6: Assortativity",
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                violations_after={k: violations_after.get(k, 0.0)
                                   for k in ["assortativity"]},
                inheritance_check=inh,
                elapsed_sec=time.time() - t6,
            ))

        STAGE_ORDER = [
            ("clustering",     "stage2", Stage2Clustering,   "Stage 2: Clustering"),
            ("avg_path_length","stage3", Stage3PathLength,   "Stage 3: Path Length"),
            ("modularity",     "stage4", Stage4Community,    "Stage 4: Modularity"),
            ("spectral_gap",   "stage5", Stage5SpectralGap,  "Stage 5: Spectral Gap"),
            ("assortativity",  "stage6", Stage6Assortativity,"Stage 6: Assortativity"),
        ]
        for pass_idx in range(1, self.n_passes):
            violations_now = constraint_violation(
                compute_metrics(G, list(targets.keys())), targets, tolerances
            )
            violated = [k for k, v in violations_now.items() if v > 1e-9]
            if not violated:
                break
            if self.verbose:
                logger.info(
                    f"  [Pass {pass_idx+1}/{self.n_passes}] "
                    f"re-running violated stages: {violated}"
                )
            for constraint_key, cfg_key, StageClass, stage_label in STAGE_ORDER:
                if constraint_key not in violated:
                    continue
                if getattr(self.spec, constraint_key, None) is None:
                    continue
                tp = time.time()
                stage = StageClass(
                    self.spec,
                    **self.stage_configs.get(cfg_key, {}),
                    verbose=self.verbose,
                )
                G = stage.run(G)
                metrics_after = compute_metrics(G, list(targets.keys()))
                violations_after = constraint_violation(metrics_after, targets, tolerances)
                report.stage_reports.append(StageReport(
                    stage_name=f"[Pass {pass_idx+1}] {stage_label}",
                    metrics_before={},
                    metrics_after=metrics_after,
                    violations_after={constraint_key: violations_after.get(constraint_key, 0.0)},
                    inheritance_check={},
                    elapsed_sec=time.time() - tp,
                ))

        final_metrics = compute_metrics(G, list(targets.keys()))
        final_violations = constraint_violation(final_metrics, targets, tolerances)
        report.final_metrics = final_metrics
        report.final_violations = final_violations
        report.satisfied = all(v <= 1e-9 for v in final_violations.values())
        report.total_elapsed_sec = time.time() - t0

        self.last_report = report

        if self.verbose:
            print(report.summary())

        return G

    def generate_batch(self, n: int) -> list[nx.Graph]:
        graphs = []
        for i in range(n):
            logger.info(f"Generating network {i+1}/{n}")
            G = self.generate()
            graphs.append(G)
        return graphs
