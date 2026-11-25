"""Reader for a specified CRCNS ret-1 recording."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio
from experiments import realdata_pipeline as rp


def find_ret1_file(raw_root: Path, file_name: str) -> Path:
    direct = raw_root / "crcns_ret-1" / "Data" / file_name
    if direct.exists():
        return direct
    matches = sorted(raw_root.glob(f"**/{file_name}"))
    matches = [
        m for m in matches if "__MACOSX" not in m.parts and not m.name.startswith("._")
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {file_name} under {raw_root}; expected "
            "data/raw/crcns_ret-1/Data/<file>.mat"
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
    return np.sort(arr[arr >= 0])


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
        raise ValueError(f"Expected 2D ret-1 spikes array, got {spikes.shape}")
    rows, cols = spikes.shape

    if ncell is not None and rows == ncell:
        if not 0 <= record_index < cols:
            raise IndexError(record_index)
        trains = [clean_train(spikes[i, record_index]) for i in range(rows)]
        orientation = "rows=cells, cols=records"
    elif ncell is not None and cols == ncell:
        if not 0 <= record_index < rows:
            raise IndexError(record_index)
        trains = [clean_train(spikes[record_index, j]) for j in range(cols)]
        orientation = "rows=records, cols=cells"
    elif nrec is not None and cols == nrec:
        if not 0 <= record_index < cols:
            raise IndexError(record_index)
        trains = [clean_train(spikes[i, record_index]) for i in range(rows)]
        orientation = "rows=cells, cols=records inferred by nrec"
    elif nrec is not None and rows == nrec:
        if not 0 <= record_index < rows:
            raise IndexError(record_index)
        trains = [clean_train(spikes[record_index, j]) for j in range(cols)]
        orientation = "rows=records, cols=cells inferred by nrec"
    else:
        raise ValueError(
            f"Cannot infer ret-1 orientation: shape={spikes.shape}, "
            f"Ncell={ncell}, Nrecord={nrec}"
        )

    if sum(len(t) > 0 for t in trains) < 2:
        raise ValueError("Too few nonempty ret-1 spike trains")
    return mat_path, datainfo, spikes.shape, orientation, trains, list(range(len(trains)))
