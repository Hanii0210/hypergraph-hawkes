"""
Formal real-data analysis: PVC-3 area17 window stability.

This script is the cleaned version of exp16_pvc3_window_stability.py.
It uses experiments/realdata_pipeline.py for the shared protocol.

Protocol:
    - load CRCNS PVC-3 area17 spontaneous spike trains
    - choose a fixed top-N neuron subset over the full analysis span
    - for each non-overlapping window and each top_m candidate setting:
        * split the window into selection and inference halves
        * select candidate hyperedges on the first half
        * fit pairwise-only and HTH models on the second half
        * report candidate-count BIC difference and exposure diagnostics

Primary evidence:
    bicdiff_candidate_count > 0 favours HTH.

Example:
    python experiments/realdata_pvc3.py --raw-root data/raw --top-m-list 1,2,3 --n-iter 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata_pipeline as rp


def find_area17_dir(raw_root: Path) -> Path:
    matches = sorted(raw_root.glob("**/spont_activity/spike_data_area17"))
    matches = [m for m in matches if "__MACOSX" not in m.parts]
    if not matches:
        raise FileNotFoundError(
            f"Could not find PVC-3 area17 under {raw_root}. "
            "Expected **/spont_activity/spike_data_area17"
        )
    return matches[0]


def load_area17(raw_root: Path):
    area_dir = find_area17_dir(raw_root)
    trains = []
    files = []

    for spk in sorted(area_dir.glob("t*.spk")):
        if spk.name.startswith("._"):
            continue
        # CRCNS pvc-3 .spk files are uint64 microsecond ticks.
        t = np.fromfile(spk, dtype=np.uint64).astype(float) / 1_000_000.0
        t = np.sort(t[np.isfinite(t)])
        trains.append(t)
        files.append(spk.name)

    if not trains:
        raise FileNotFoundError(f"No .spk files found in {area_dir}")

    return area_dir, trains, files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--starts", default="0,20,40,60,80")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--top-m-list", default="1,2,3")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--lambda-l1", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out-prefix", default="experiments/results/realdata/realdata_pvc3_area17")
    parser.add_argument("--verbose-hth", action="store_true")
    args = parser.parse_args()

    starts = rp.parse_float_list(args.starts)
    top_m_list = rp.parse_int_list(args.top_m_list)

    span_start = min(starts)
    span_end = max(s + args.duration for s in starts)

    area_dir, trains, files = load_area17(Path(args.raw_root))

    selected_indices, mapping = rp.select_fixed_subset(
        trains=trains,
        top_n=args.top_n,
        span_start=span_start,
        span_end=span_end,
        original_labels=files,
        label_key="original_file",
    )

    print("=" * 110)
    print("Formal real-data analysis: PVC-3 area17")
    print("=" * 110)
    print(f"source_dir      : {area_dir}")
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
            f"  node {m['new_node']:2d} <- {m['original_file']}, "
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
        dataset="PVC-3 area17",
        beta=args.beta,
        delta=args.delta,
        n_iter=args.n_iter,
        lambda_l1=args.lambda_l1,
        seed=args.seed,
        label_key="original_file",
        extra={
            "data_type": "continuous spike-time",
            "source": str(area_dir),
        },
        verbose_hth=args.verbose_hth,
        print_progress=True,
    )

    out_prefix = Path(args.out_prefix)
    csv_path = out_prefix.with_suffix(".csv")
    pkl_path = out_prefix.with_suffix(".pkl")
    json_path = out_prefix.with_name(out_prefix.name + "_mapping.json")

    rp.write_rows_csv(rows, csv_path)
    rp.save_pickle(
        {
            "args": vars(args),
            "source_dir": str(area_dir),
            "mapping": mapping,
            "rows": rows,
            "results": results,
        },
        pkl_path,
    )
    rp.save_json(
        {
            "source_dir": str(area_dir),
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
