"""Reader for CRCNS PVC-11 spontaneous spike trains."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio
from experiments import realdata_pipeline as rp


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
    matches = [
        m for m in matches if "__MACOSX" not in m.parts and not m.name.startswith("._")
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {pattern} under {raw_root}; extract pvc-11 first."
        )
    return matches[0]


def load_pvc11_spont(raw_root: Path, monkey: int):
    mat_path = find_pvc11_file(raw_root, monkey)
    mat = sio.loadmat(mat_path, simplify_cells=True)
    data = mat.get("data", {})
    if not isinstance(data, dict) or "EVENTS" not in data:
        raise KeyError(f"Could not find data['EVENTS'] in {mat_path}")
    raw_trains = [rp.as_1d_float(x) for x in iter_event_list(data["EVENTS"])]
    kept = [(idx, train) for idx, train in enumerate(raw_trains) if len(train) > 0]
    if not kept:
        raise ValueError(f"No nonempty spike trains in {mat_path}")
    return mat_path, [train for _, train in kept], [idx for idx, _ in kept]
