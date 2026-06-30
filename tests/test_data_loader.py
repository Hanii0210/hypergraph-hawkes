import sys
sys.path.insert(0, ".")

import os
import tempfile
from simulation.data_loader import (
    load_events_from_csv,
    save_events_to_csv,
    summarise_events,
)


# =============================================================================
# Test 1: round-trip save and load
# =============================================================================
print("--- Test 1: save and load round-trip ---")
events_in = [(0.1, 0), (0.5, 2), (1.2, 1), (1.8, 0), (2.4, 2)]

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
tmp.close()
save_events_to_csv(events_in, tmp.name)

events_out, n_nodes, T = load_events_from_csv(tmp.name, T=3.0)
print(f"  events_out : {events_out}")
print(f"  n_nodes    : {n_nodes}")
print(f"  T          : {T}")

assert len(events_in) == len(events_out), "FAILED: count mismatch"
for (t1, n1), (t2, n2) in zip(events_in, events_out):
    assert abs(t1 - t2) < 1e-5, "FAILED: time mismatch"
    assert n1 == n2, "FAILED: node mismatch"
assert n_nodes == 3, "FAILED: n_nodes inference"
print("PASSED\n")

os.unlink(tmp.name)


# =============================================================================
# Test 2: summarise_events
# =============================================================================
print("--- Test 2: summarise ---")
summarise_events(events_in)
print("PASSED\n")


# =============================================================================
# Test 3: load with auto-sorting
# =============================================================================
print("--- Test 3: auto-sort ---")
events_unsorted = [(2.0, 0), (0.5, 1), (1.5, 0)]

tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
tmp.close()
save_events_to_csv(events_unsorted, tmp.name)

events_loaded, _, _ = load_events_from_csv(tmp.name, T=3.0, sort=True)
print(f"  loaded sorted : {events_loaded}")
times = [t for t, _ in events_loaded]
assert times == sorted(times), "FAILED: not sorted"
print("PASSED\n")

os.unlink(tmp.name)


# =============================================================================
# Test 4: error on missing columns
# =============================================================================
print("--- Test 4: error on bad CSV ---")
bad_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
bad_csv.write("foo,bar\n1,2\n")
bad_csv.close()

try:
    load_events_from_csv(bad_csv.name, T=3.0)
    assert False, "FAILED: should have raised"
except ValueError as e:
    print(f"  caught expected error: {e}")
    print("PASSED\n")

os.unlink(bad_csv.name)

print("=== All tests passed ===")
