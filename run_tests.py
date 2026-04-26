import subprocess
import sys
import time
import os

# =============================================================================
# Unified test runner: runs all unit tests and reports pass/fail summary
# =============================================================================

TESTS_DIR = "tests"
TESTS = [
    "test_kernel.py",
    "test_tensor.py",
    "test_estep.py",
    "test_mstep.py",
    "test_simulator.py",
    "test_candidate_filter.py",
    "test_data_loader.py",
]


def run_one(path):
    start = time.time()
    # Run from the project root so that imports of models/ etc. resolve
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    elapsed = time.time() - start
    passed = (result.returncode == 0)
    return passed, elapsed, result.stdout, result.stderr


def main():
    print("=" * 70)
    print("Running all unit tests")
    print("=" * 70)

    results = []
    for test in TESTS:
        path = os.path.join(TESTS_DIR, test)
        print(f"\n>>> {test}")
        passed, elapsed, stdout, stderr = run_one(path)

        if passed:
            n_passed = stdout.count("PASSED")
            print(f"    OK    ({n_passed} sub-tests, {elapsed:.1f}s)")
        else:
            print(f"    FAIL  ({elapsed:.1f}s)")
            print("    --- stderr ---")
            print(stderr[-500:])

        results.append((test, passed, elapsed))

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    n_pass = sum(1 for _, p, _ in results if p)
    n_total = len(results)
    total_time = sum(t for _, _, t in results)

    for test, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {test:<35}  {elapsed:>5.1f}s")

    print("-" * 70)
    print(f"  {n_pass} / {n_total} test files passed,  total {total_time:.1f}s")

    if n_pass == n_total:
        print("\n  All tests passed.")
        sys.exit(0)
    else:
        print(f"\n  {n_total - n_pass} test file(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()