"""
Formal real-data analysis: ret-1 retinal recording window stability.

This is the fixed ret-1 loader version. It deliberately avoids generic recursive
MATLAB parsing because ret-1 stores spikes as a clear Ncell x Nrecord object array.

For 20080516_R1.mat:
    datainfo.Ncell = 7
    datainfo.RecNo = [4, 5, 6]
    spikes.shape   = (7, 3)

The selected record is therefore:
    trains = spikes[:, record_index]

Primary evidence:
    bicdiff_candidate_count > 0 favours HTH.

Example:
    python experiments/real01_ret1.py --raw-root data/raw --file 20080516_R1.mat --record-index 0 --top-m-list 1,2,3 --n-iter 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import realdata_pipeline as rp


def find_ret1_file(raw_root: Path, file_name: str) -> Path:
    direct = raw_root / "crcns_ret-1" / "Data" / file_name
    if direct.exists():
        return direct

    matches = sorted(raw_root.glob(f"**/{file_name}"))
    matches = [m for m in matches if "__MACOSX" not in m.parts and not m.name.startswith("._")]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {file_name} under {raw_root}. "
            "Expected data/raw/crcns_ret-1/Data/<file>.mat"
        )
    return matches[0]


def datainfo_get_int(datainfo, key, default=None):
    if not isinstance(datainfo, dict) or key not in datainfo:
        return default
    try:
        return int(np.asarray(datainfo[key]).ravel()[0])
    except Exception:
        try:
            return int(datainfo[key])
        except Exception:
            return default


def datainfo_rec_count(datainfo):
    if not isinstance(datainfo, dict) or "RecNo" not in datainfo:
        return None
    try:
        return int(np.asarray(datainfo["RecNo"]).ravel().size)
    except Exception:
        try:
            return len(datainfo["RecNo"])
        except Exception:
            return None


def clean_train(x):
    arr = rp.as_1d_float(x)
    arr = arr[arr >= 0]
    return np.sort(arr)


def load_ret1_record(raw_root: Path, file_name: str, record_index: int):
    mat_path = find_ret1_file(raw_root, file_name)
    mat = sio.loadmat(mat_path, simplify_cells=True)

    if "spikes" not in mat:
        raise KeyError(f"No variable named 'spikes' in {mat_path}")

    datainfo = mat.get("datainfo", mat.get("DataInfo", None))
    ncell = datainfo_get_int(datainfo, "Ncell", default=None)
    nrec = datainfo_rec_count(datainfo)

    spikes = np.squeeze(np.asarray(mat["spikes"], dtype=object))

    if spikes.ndim != 2:
        raise ValueError(f"Expected ret-1 spikes to be 2D after squeeze, got shape={spikes.shape}")

    r, c = spikes.shape

    # Correct default structure: rows=cells, columns=records.
    if ncell is not None and r == ncell:
        if not (0 <= record_index < c):
            raise IndexError(f"record_index={record_index} out of range for spikes.shape={spikes.shape}")
        trains = [clean_train(spikes[i, record_index]) for i in range(r)]
        orientation = "rows=cells, cols=records"

    # Fallback for transposed files.
    elif ncell is not None and c == ncell:
        if not (0 <= record_index < r):
            raise IndexError(f"record_index={record_index} out of range for spikes.shape={spikes.shape}")
        trains = [clean_train(spikes[record_index, j]) for j in range(c)]
        orientation = "rows=records, cols=cells"

    # Last fallback: use nrec if available.
    elif nrec is not None and c == nrec:
        if not (0 <= record_index < c):
            raise IndexError(f"record_index={record_index} out of range for spikes.shape={spikes.shape}")
        trains = [clean_train(spikes[i, record_index]) for i in range(r)]
        orientation = "rows=cells, cols=records inferred by nrec"

    elif nrec is not None and r == nrec:
        if not (0 <= record_index < r):
            raise IndexError(f"record_index={record_index} out of range for spikes.shape={spikes.shape}")
        trains = [clean_train(spikes[record_index, j]) for j in range(c)]
        orientation = "rows=records, cols=cells inferred by nrec"

    else:
        raise ValueError(
            f"Cannot infer ret-1 spikes orientation. shape={spikes.shape}, "
            f"Ncell={ncell}, Nrecord={nrec}"
        )

    nonempty = sum(len(t) > 0 for t in trains)
    if nonempty < 2:
        raise ValueError(f"Extracted too few nonempty trains: {nonempty}")

    original_labels = list(range(len(trains)))
    return mat_path, datainfo, spikes.shape, orientation, trains, original_labels


def print_datainfo(datainfo):
    if datainfo is None:
        return
    print("\nDatainfo summary:")
    if isinstance(datainfo, dict):
        for k, v in datainfo.items():
            if str(k).startswith("__"):
                continue
            try:
                vv = v.tolist() if isinstance(v, np.ndarray) else v
            except Exception:
                vv = "<unprintable>"
            print(f"  {k}: {vv}")
    else:
        print(f"  {datainfo}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--file", default="20080516_R1.mat")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--starts", default="0,20,40,60,80")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--top-n", type=int, default=7)
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

    mat_path, datainfo, spikes_shape, orientation, trains, original_indices = load_ret1_record(
        Path(args.raw_root), args.file, args.record_index
    )

    if args.top_n > len(trains):
        raise ValueError(f"--top-n={args.top_n} but only {len(trains)} cells were loaded.")

    selected_indices, mapping = rp.select_fixed_subset(
        trains=trains,
        top_n=args.top_n,
        span_start=span_start,
        span_end=span_end,
        original_labels=original_indices,
        label_key="original_cell",
    )

    print("=" * 110)
    print("Formal real-data analysis: ret-1 retinal recording")
    print("=" * 110)
    print(f"source_file     : {mat_path}")
    print(f"spikes shape    : {spikes_shape}")
    print(f"orientation     : {orientation}")
    print(f"record_index    : {args.record_index}")
    print(f"n_loaded_cells  : {len(trains)}")
    print(f"analysis span   : [{span_start}, {span_end}) sec")
    print(f"starts          : {starts}")
    print(f"duration        : {args.duration}")
    print(f"top_n fixed     : {args.top_n}")
    print(f"top_m_list      : {top_m_list}")
    print(f"beta/delta      : {args.beta} / {args.delta}")
    print(f"n_iter/lambda   : {args.n_iter} / {args.lambda_l1}")

    print_datainfo(datainfo)

    print("\nFixed cell mapping:")
    for m in mapping:
        print(
            f"  node {m['new_node']:2d} <- original cell {m['original_cell']}, "
            f"count_span={m['count_in_analysis_span']}"
        )

    print("\nRunning window-stability protocol...")
    print("-" * 110)

    dataset_name = f"ret-1 {Path(args.file).stem} rec{args.record_index}"
    rows, results = rp.run_window_grid(
        trains=trains,
        selected_indices=selected_indices,
        mapping=mapping,
        starts=starts,
        duration=args.duration,
        top_n=args.top_n,
        top_m_list=top_m_list,
        dataset=dataset_name,
        beta=args.beta,
        delta=args.delta,
        n_iter=args.n_iter,
        lambda_l1=args.lambda_l1,
        seed=args.seed,
        label_key="original_cell",
        extra={
            "data_type": "continuous spike-time",
            "source": str(mat_path),
            "record_index": int(args.record_index),
            "spikes_shape": str(spikes_shape),
            "orientation": orientation,
        },
        verbose_hth=args.verbose_hth,
        print_progress=True,
    )

    out_prefix = Path(args.out_prefix) if args.out_prefix else Path(
        f"experiments/results/realdata/real01_ret1_{Path(args.file).stem}_rec{args.record_index}"
    )
    csv_path = out_prefix.with_suffix(".csv")
    pkl_path = out_prefix.with_suffix(".pkl")
    json_path = out_prefix.with_name(out_prefix.name + "_mapping.json")

    rp.write_rows_csv(rows, csv_path)
    rp.save_pickle(
        {
            "args": vars(args),
            "source_file": str(mat_path),
            "spikes_shape": str(spikes_shape),
            "orientation": orientation,
            "mapping": mapping,
            "rows": rows,
            "results": results,
        },
        pkl_path,
    )
    rp.save_json(
        {
            "source_file": str(mat_path),
            "spikes_shape": str(spikes_shape),
            "orientation": orientation,
            "record_index": int(args.record_index),
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
