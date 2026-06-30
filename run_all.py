"""
End-to-end experiment pipeline.

Runs all 10 experiments in order, then regenerates all 9 figures.
Total runtime: approximately 60-90 minutes on a single CPU.

Usage:
    python run_all.py
"""

import subprocess
import sys
import time
import os


EXPERIMENTS = [
    ("Exp 1   Recovery demo (single seed)",      "experiments/exp1_demo.py"),
    ("Exp 1b  Recovery robustness (25 seeds)",    "experiments/exp1b_recovery_robustness.py"),
    ("Exp 2   Regularisation path",               "experiments/exp2_regpath.py"),
    ("Exp 3   Convergence (20 random inits)",      "experiments/exp3_convergence.py"),
    ("Exp 4   Phase transition",                   "experiments/exp4_phase.py"),
    ("Exp 5   Copula validation",                  "experiments/exp5_copula.py"),
    ("Exp 6   3-node hyperedge",                   "experiments/exp6_3node_hyperedge.py"),
    ("Exp 7   Likelihood gap (falsifiability)",    "experiments/exp7_likelihood_gap.py"),
    ("Exp 8   Delta sensitivity",                  "experiments/exp8_delta_sensitivity.py"),
    ("Exp 9   Scalability",                        "experiments/exp9_scalability.py"),
    ("Exp 10  Real neural data (CRCNS ret-1)",     "experiments/exp10_realdata.py"),
    ("Exp 11  Bias ablation (beta sweep)",          "experiments/exp11_bias_ablation.py"),
    ("Exp 12a Generate CP-rank data (cached)",      "experiments/exp12_generate_data.py"),
    ("Exp 12b CP rank selection sweep",             "experiments/exp12_rank_sweep.py"),
    ("Exp 13a Calibration (pairwise null)",         "experiments/exp13_calibration.py"),
    ("Exp 13b Calibration CONTROL (poisson)",       "experiments/exp13_calibration.py", ["40", "700", "poisson"]),
    ("Exp 14  Identification diagnostic",           "experiments/exp14_identification_diagnostic.py"),
    ("Exp 15  Interaction baseline",                "experiments/exp15_interaction_baseline.py"),
]

PLOTS = [
    ("Plot 1b  Recovery robustness",    "experiments/exp1b_plot.py"),
    ("Plot 2   Regularisation path",    "experiments/exp2_plot.py"),
    ("Plot 3   Convergence",            "experiments/exp3_plot.py"),
    ("Plot 4   Phase transition",       "experiments/exp4_plot.py"),
    ("Plot 5   Copula",                 "experiments/exp5_plot.py"),
    ("Plot 6   3-node hyperedge",       "experiments/exp6_plot.py"),
    ("Plot 7   Likelihood gap",         "experiments/exp7_plot.py"),
    ("Plot 8   Delta sensitivity",      "experiments/exp8_plot.py"),
    ("Plot 9   Scalability",            "experiments/exp9_plot.py"),
    ("Plot 10  Real neural data",                   "experiments/exp10_plot.py"),
    ("Plot 11  Bias ablation",                       "experiments/exp11_plot.py"),
    ("Plot 12  CP rank selection",                   "experiments/exp12_plot.py"),
    ("Plot 13  Calibration",            "experiments/exp13_plot.py"),
    ("Plot 14  Identification diag.",   "experiments/exp14_plot.py"),
    ("Plot 15  Interaction baseline",   "experiments/exp15_plot.py"),
]


def run_script(label, path, args=None):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*60}")
    start = time.time()
    result = subprocess.run(
        [sys.executable, path] + (args or []),
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env={**os.environ, "MPLBACKEND": "Agg"},
    )
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"  [{status}]  {elapsed:.1f}s")
    return result.returncode == 0, elapsed


def main():
    print("=" * 60)
    print("  Hypergraph Hawkes: Full Experiment Pipeline")
    print("=" * 60)

    total_start = time.time()
    results = []

    # Phase 1: run all experiments
    print("\n\n>>> PHASE 1: Running experiments <<<\n")
    for label, path, *rest in EXPERIMENTS:
        ok, t = run_script(label, path, rest[0] if rest else None)
        results.append((label, ok, t))

    # Phase 2: generate all figures (non-interactive backend)
    print("\n\n>>> PHASE 2: Generating figures <<<\n")
    for label, path, *rest in PLOTS:
        ok, t = run_script(label, path, rest[0] if rest else None)
        results.append((label, ok, t))

    # Summary
    total_time = time.time() - total_start
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)

    print("\n\n" + "=" * 60)
    print("  Pipeline Summary")
    print("=" * 60)
    for label, ok, t in results:
        s = "PASS" if ok else "FAIL"
        print(f"  [{s}]  {label:<45} {t:>6.1f}s")
    print("-" * 60)
    print(f"  {n_pass} / {n_total} steps completed")
    print(f"  Total wall-clock time: {total_time/60:.1f} minutes")

    if n_pass == n_total:
        print("\n  All steps passed.")
    else:
        print(f"\n  {n_total - n_pass} step(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()