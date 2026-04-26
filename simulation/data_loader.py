import csv
import numpy as np


def load_events_from_csv(
    path: str,
    time_col: str = "time",
    node_col: str = "node",
    sort: bool = True,
):
    """
    Load events from a CSV file.

    Expected format:
        time,node
        0.123,0
        0.456,2
        0.789,1
        ...

    Parameters
    ----------
    path     : str    path to CSV file
    time_col : str    column name for event times
    node_col : str    column name for node ids
    sort     : bool   sort events by time (default True)

    Returns
    -------
    events  : list of (float time, int node) tuples
    n_nodes : int, inferred from max node id + 1
    T       : float, inferred from max time
    """
    events = []
    max_node = -1
    max_time = 0.0

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if time_col not in reader.fieldnames or node_col not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain columns '{time_col}' and '{node_col}'. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            t    = float(row[time_col])
            node = int(row[node_col])
            events.append((t, node))
            if node > max_node:
                max_node = node
            if t > max_time:
                max_time = t

    if sort:
        events.sort(key=lambda x: x[0])

    n_nodes = max_node + 1
    T = max_time
    return events, n_nodes, T


def save_events_to_csv(events, path, time_col="time", node_col="node"):
    """Write a list of (time, node) tuples to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([time_col, node_col])
        for t, node in events:
            writer.writerow([f"{t:.6f}", int(node)])


def summarise_events(events, n_nodes=None, T=None):
    """Print a quick summary of an event stream."""
    if len(events) == 0:
        print("  (empty event stream)")
        return

    times = [t for t, _ in events]
    nodes = [n for _, n in events]

    if n_nodes is None:
        n_nodes = max(nodes) + 1
    if T is None:
        T = max(times)

    counts = [0] * n_nodes
    for n in nodes:
        counts[n] += 1

    print(f"  total events  : {len(events)}")
    print(f"  time span     : [{min(times):.3f}, {max(times):.3f}]   T={T}")
    print(f"  n_nodes       : {n_nodes}")
    print(f"  events/node   : {counts}")
    print(f"  rate (per T)  : {len(events) / T:.3f}")