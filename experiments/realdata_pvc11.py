"""
Formal real-data analysis: PVC-11 spontaneous window stability.

This script is the cleaned version of exp17_pvc11_window_stability.py.
It uses experiments/realdata_pipeline.py for the shared protocol.

Default target:
    PVC-11 spontaneous monkey 2, because it was the smallest spontaneous PVC-11
    file in the inventory and is suitable for the robustness analysis.

Protocol:
    - load PVC-11 spontaneous spike trains
    - choose a fixed top-N neuron subset over the full analysis span
    - for each non-overlapping window and each top_m candidate setting:
        * split the window into selection and inference halves
        * select candidate hyperedges on the first half
        * fit pairwise-only and HTH models on the second half
        * report candidate-count BIC difference and exposure diagnostics

Primary evidence:
    bicdiff_candidate_count > 0 favours HTH.

Example:
    python experiments/realdata_pvc11.py --raw-root data/raw --monkey 2 --top-m-list 1,2,3 --n-iter 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata_pipeline as rp


def iter_event_list(events):
    if isinstance(events, np.ndarray):
        if events.dtype == object:
            return [events.flat[i] for i in range(events.size)]
        if events.ndim == 1:
            return [events]
    if isinstance(events, (list, tuple)):
        return list(events)
    return []


def find_pvc11_file(raw_root: Path, monkey: int) -> Path:
    pattern = f"**/data_and_scripts/spikes_spontaneous/spiketimesmonkey{monkey}spont.mat"
    matches = sorted(raw_root.glob(pattern))
    matches = [m for m in matches if "__MACOSX" not in m.parts and not m.name.startswith("._")]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {pattern} under {raw_root}. "
            "Check that pvc-11 data_and_scripts.tar.gz has been extracted."
        )
    return matches[0]


def load_pvc11_spont(raw_root: Path, monkey: int):
    mat_path = find_pvc11_file(raw_root, monkey)
    mat = sio.loadmat(mat_path, simplify_cells=True)
    data = mat.get("data", {})

    if not isinstance(data, dict) or "EVENTS" not in data:
        raise KeyError(
            f"Could not find data['EVENTS'] in {mat_path}. "
            f"Available top-level keys: {list(mat.keys())}"
        )

    raw_trains = [rp.as_1d_float(x) for x in iter_event_list(data["EVENTS"])]

    keep = [(idx, t) for idx, t in enumerate(raw_trains) if len(t) > 0]
    original_indices = [idx for idx, _ in keep]
    trains = [t for _, t in keep]

    if not trains:
        raise ValueError(f"No nonempty spike trains found in {mat_path}")

    return mat_path, trains, original_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--monkey", type=int, default=2)
    parser.add_argument("--starts", default="0,40,80,120,160")
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-m-list", default="1,2,3")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-prefix", default=None)
    parser.add_argument("--verbose-hth", action="store_true")
    args = parser.parse_args()

    starts = rp.parse_float_list(args.starts)
    top_m_list = rp.parse_int_list(args.top_m_list)

    span_start = min(starts)
    span_end = max(s + args.duration for s in starts)

    mat_path, trains, original_indices = load_pvc11_spont(Path(args.raw_root), args.monkey)

    selected_indices, mapping = rp.select_fixed_subset(
        trains=trains,
        top_n=args.top_n,
        span_start=span_start,
        span_end=span_end,
        original_labels=original_indices,
        label_key="original_index",
    )

    print("=" * 110)
    print("Formal real-data analysis: PVC-11 spontaneous")
    print("=" * 110)
    print(f"source_file     : {mat_path}")
    print(f"monkey          : {args.monkey}")
    print(f"analysis span   : [{span_start}, {span_end}) sec")
    print(f"starts          : {starts}")
    print(f"duration        : {args.duration}")
    print(f"top_n fixed     : {args.top_n}")
    print(f"top_m_list      : {top_m_list}")
    print(f"beta/delta      : {args.beta} / {args.delta}")
    print(f"n_iter/lambda   : {args.n_iter} / {args.lambda_l1}")
    print("\nFixed neuron mapping:")
    for m in mapping:
        print(
            f"  node {m['new_node']:2d} <- original {m['original_index']:3d}, "
            f"count_span={m['count_in_analysis_span']}"
        )

    print("\nRunning window-stability protocol...")
    print("-" * 110)

    rows, results = rp.run_window_grid(
        trains=trains,
        selected_indices=selected_indices,
        mapping=mapping,
        starts=starts,
        duration=args.duration,
        top_n=args.top_n,
        top_m_list=top_m_list,
        dataset=f"PVC-11 monkey{args.monkey}",
        beta=args.beta,
        delta=args.delta,
        n_iter=args.n_iter,
        lambda_l1=args.lambda_l1,
        seed=args.seed,
        label_key="original_index",
        extra={
            "data_type": "continuous spike-time",
            "source": str(mat_path),
            "condition": "spontaneous",
            "monkey": int(args.monkey),
        },
        verbose_hth=args.verbose_hth,
        print_progress=True,
    )

    out_prefix = Path(args.out_prefix) if args.out_prefix else Path(
        f"experiments/results/realdata/realdata_pvc11_monkey{args.monkey}"
    )
    csv_path = out_prefix.with_suffix(".csv")
    pkl_path = out_prefix.with_suffix(".pkl")
    json_path = out_prefix.with_name(out_prefix.name + "_mapping.json")

    rp.write_rows_csv(rows, csv_path)
    rp.save_pickle(
        {
            "args": vars(args),
            "source_file": str(mat_path),
            "mapping": mapping,
            "rows": rows,
            "results": results,
        },
        pkl_path,
    )
    rp.save_json(
        {
            "source_file": str(mat_path),
            "analysis_span": [span_start, span_end],
            "mapping": mapping,
        },
        json_path,
    )

    print("-" * 110)
    print(f"Saved CSV    : {csv_path}")
    print(f"Saved PKL    : {pkl_path}")
    print(f"Saved mapping: {json_path}")

    rp.print_bic_summary(rows)


if __name__ == "__main__":
    main()
