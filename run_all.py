"""
End-to-end experiment runner for the Hyperedge-triggered Hawkes project.

Default behaviour:
    1. Run the full test suite.
    2. Run synthetic experiments 1-15, excluding the legacy exp10 real-data script.
    3. Regenerate synthetic figures.
    4. Run the formal real-data scripts for ret-1, PVC-3, and PVC-11.
    5. Regenerate real-data summary figures.

Usage examples:
    python run_all.py
    python run_all.py --skip-realdata
    python run_all.py --skip-synthetic
    python run_all.py --quick
    python run_all.py --continue-on-error

Notes:
    - The legacy archive/legacy_realdata_scripts/exp10_realdata.py is intentionally not used here.
      Formal real-data analysis is handled by:
          experiments/realdata_ret1.py
          experiments/realdata_pvc3.py
          experiments/realdata_pvc11.py
          experiments/plot_realdata_summary.py
    - Candidate-count BIC is the primary real-data model-comparison statistic.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Step:
    label: str
    command: List[str]
    required: bool = True


SYNTHETIC_EXPERIMENTS: List[Step] = [
    Step("Syn 1  recovery and robustness", [sys.executable, "experiments/syn01_recovery_robustness.py"]),
    Step("Syn 2  regularization path", [sys.executable, "experiments/syn02_regularization_path.py"]),
    Step("Syn 3  EM convergence", [sys.executable, "experiments/syn03_em_convergence.py"]),
    Step("Syn 4  strength sensitivity", [sys.executable, "experiments/syn04_strength_sensitivity.py"]),
    Step("Syn 5  likelihood separation", [sys.executable, "experiments/syn05_likelihood_separation.py"]),
    Step("Syn 6  trigger-window sensitivity", [sys.executable, "experiments/syn06_trigger_window_sensitivity.py"]),
    Step("Syn 7  scalability", [sys.executable, "experiments/syn07_scalability.py"]),
    Step("Syn 8  bias ablation", [sys.executable, "experiments/syn08_bias_ablation.py"]),
    Step("Syn 9  identification diagnostic", [sys.executable, "experiments/syn09_identification_diagnostic.py"]),
    Step("Syn 10 interaction baseline", [sys.executable, "experiments/syn10_interaction_baseline.py"]),
]


SYNTHETIC_PLOTS: List[Step] = [
    Step("Plot Syn 1  recovery and robustness", [sys.executable, "experiments/syn01_plot_recovery_robustness.py"]),
    Step("Plot Syn 2  regularization path", [sys.executable, "experiments/syn02_plot_regularization_path.py"]),
    Step("Plot Syn 3  EM convergence", [sys.executable, "experiments/syn03_plot_em_convergence.py"]),
    Step("Plot Syn 4  strength sensitivity", [sys.executable, "experiments/syn04_plot_strength_sensitivity.py"]),
    Step("Plot Syn 5  likelihood separation", [sys.executable, "experiments/syn05_plot_likelihood_separation.py"]),
    Step("Plot Syn 6  trigger-window sensitivity", [sys.executable, "experiments/syn06_plot_trigger_window_sensitivity.py"]),
    Step("Plot Syn 7  scalability", [sys.executable, "experiments/syn07_plot_scalability.py"]),
    Step("Plot Syn 8  bias ablation", [sys.executable, "experiments/syn08_plot_bias_ablation.py"]),
    Step("Plot Syn 9  identification diagnostic", [sys.executable, "experiments/syn09_plot_identification_diagnostic.py"]),
    Step("Plot Syn 10 interaction baseline", [sys.executable, "experiments/syn10_plot_interaction_baseline.py"]),
]


def realdata_steps(raw_root: str, top_m_list: str, n_iter: int) -> List[Step]:
    n_iter_s = str(n_iter)
    return [
        Step(
            "Real data ret-1 formal window stability",
            [
                sys.executable,
                "experiments/realdata_ret1.py",
                "--raw-root",
                raw_root,
                "--file",
                "20080516_R1.mat",
                "--record-index",
                "0",
                "--top-m-list",
                top_m_list,
                "--n-iter",
                n_iter_s,
            ],
        ),
        Step(
            "Real data PVC-3 area17 formal window stability",
            [
                sys.executable,
                "experiments/realdata_pvc3.py",
                "--raw-root",
                raw_root,
                "--top-m-list",
                top_m_list,
                "--n-iter",
                n_iter_s,
            ],
        ),
        Step(
            "Real data PVC-11 monkey2 formal window stability",
            [
                sys.executable,
                "experiments/realdata_pvc11.py",
                "--raw-root",
                raw_root,
                "--monkey",
                "2",
                "--top-m-list",
                top_m_list,
                "--n-iter",
                n_iter_s,
            ],
        ),
    ]


REALDATA_PLOTS: List[Step] = [
    Step("Plot real-data summary figures", [sys.executable, "experiments/plot_realdata_summary.py"]),
]


def run_step(step: Step) -> tuple[bool, float]:
    print("\n" + "=" * 78)
    print(step.label)
    print(" ".join(step.command))
    print("=" * 78)

    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"

    start = time.time()
    result = subprocess.run(step.command, cwd=PROJECT_ROOT, env=env)
    elapsed = time.time() - start

    ok = result.returncode == 0
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {step.label} ({elapsed:.1f}s)")
    return ok, elapsed


def add_section(plan: List[Step], title: str, steps: Iterable[Step]) -> None:
    plan.append(Step(f"--- {title} ---", [sys.executable, "-c", "print('section')"], required=False))
    plan.extend(steps)


def build_plan(args: argparse.Namespace) -> List[Step]:
    if args.quick:
        args.skip_synthetic = True
        args.skip_realdata = True
        args.skip_plots = False

    plan: List[Step] = []

    if not args.skip_tests:
        plan.append(Step("Test suite", [sys.executable, "run_tests.py"]))

    if not args.skip_synthetic:
        plan.extend(SYNTHETIC_EXPERIMENTS)
        if not args.skip_plots:
            plan.extend(SYNTHETIC_PLOTS)

    if not args.skip_realdata:
        plan.extend(realdata_steps(args.realdata_raw_root, args.realdata_top_m_list, args.realdata_n_iter))
        if not args.skip_plots:
            plan.extend(REALDATA_PLOTS)

    if args.quick and not args.skip_plots:
        plan.extend(REALDATA_PLOTS)

    if args.include_schematic and not args.skip_plots:
        plan.append(Step("Plot HTH schematic", [sys.executable, "experiments/schematics/plot_hth_model_schematic.py"]))

    return plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HTH tests, experiments, and figures.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run tests and regenerate cached real-data summary figures only.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip run_tests.py.")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip synthetic experiments and synthetic plots.")
    parser.add_argument("--skip-realdata", action="store_true", help="Skip formal real-data scripts and real-data plots.")
    parser.add_argument("--skip-plots", action="store_true", help="Skip all plotting scripts.")
    parser.add_argument(
        "--include-schematic",
        action="store_true",
        help="Also run experiments/schematics/plot_hth_model_schematic.py. Disabled by default because the schematic is optional.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later steps after a failed step.",
    )
    parser.add_argument("--realdata-raw-root", default="data/raw", help="Root directory for raw real datasets.")
    parser.add_argument("--realdata-top-m-list", default="1,2,3", help="Comma-separated candidate counts.")
    parser.add_argument("--realdata-n-iter", type=int, default=20, help="EM iterations for formal real-data scripts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_plan(args)

    if not plan:
        print("No steps selected.")
        return

    print("=" * 78)
    print("Hyperedge-triggered Hawkes project pipeline")
    print("=" * 78)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Number of steps: {len(plan)}")
    print("Candidate-count BIC is the primary formal real-data statistic.")

    total_start = time.time()
    results: List[tuple[str, bool, float]] = []

    for step in plan:
        ok, elapsed = run_step(step)
        results.append((step.label, ok, elapsed))

        if not ok and not args.continue_on_error:
            print("\nStopping after first failure. Use --continue-on-error to continue.")
            break

    total_elapsed = time.time() - total_start
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)

    print("\n" + "=" * 78)
    print("Pipeline summary")
    print("=" * 78)
    for label, ok, elapsed in results:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label:<56} {elapsed:>8.1f}s")

    print("-" * 78)
    print(f"{n_pass} / {n_total} completed")
    print(f"Total wall-clock time: {total_elapsed / 60:.1f} minutes")

    if n_pass != n_total:
        raise SystemExit(1)

    print("\nAll selected steps passed.")


if __name__ == "__main__":
    main()
