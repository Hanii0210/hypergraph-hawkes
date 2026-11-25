"""Reader for CRCNS PVC-3 area 17 spike trains."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.io as sio
from experiments import realdata_pipeline as rp


def find_area17_dir(raw_root: Path) -> Path:
    matches = sorted(raw_root.glob("**/spont_activity/spike_data_area17"))
    matches = [m for m in matches if "__MACOSX" not in m.parts]
    if not matches:
        raise FileNotFoundError(
            f"Could not find PVC-3 area17 under {raw_root}; expected "
            "**/spont_activity/spike_data_area17"
        )
    return matches[0]


def load_area17(raw_root: Path):
    area_dir = find_area17_dir(raw_root)
    trains = []
    files = []
    for spk in sorted(area_dir.glob("t*.spk")):
        if spk.name.startswith("._"):
            continue
        ticks = np.fromfile(spk, dtype=np.uint64).astype(float) / 1_000_000.0
        trains.append(np.sort(ticks[np.isfinite(ticks)]))
        files.append(spk.name)
    if not trains:
        raise FileNotFoundError(f"No .spk files found in {area_dir}")
    return area_dir, trains, files
