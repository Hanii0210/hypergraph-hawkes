# Obtaining the neural recordings

This archive does not redistribute the CRCNS recordings. The
official dataset pages provide the download entries, documentation, citations,
and access conditions. A free CRCNS account is required; see the
[CRCNS download instructions](https://crcns.org/download).

CRCNS's [general terms](https://crcns.org/terms) restrict redistribution unless
the dataset page states different terms. Obtain the recordings from CRCNS
instead of bundling them in a public copy of this code or its result archives.

## Official sources and files used by this study

| Dataset | Dataset DOI | Official description and download | Required recording |
| --- | --- | --- | --- |
| ret-1 | [10.6080/K0RF5RZT](https://doi.org/10.6080/K0RF5RZT) | [Description](https://crcns.org/data-sets/retina/ret-1/about-ret-1); [download](https://download.crcns.org/ret-1) | `20080516_R1.mat`, record index `0` in Python |
| PVC-3 | [10.6080/K0MW2F2J](https://doi.org/10.6080/K0MW2F2J) | [Description](https://crcns.org/data-sets/vc/pvc-3/about); [download](https://download.crcns.org/pvc-3) | Spontaneous activity, area 17: all `t*.spk` files in `spont_activity/spike_data_area17` |
| PVC-11 | [10.6080/K0NC5Z4X](https://doi.org/10.6080/K0NC5Z4X) | [Description](https://crcns.org/data-sets/vc/pvc-11/about); [download](https://download.crcns.org/pvc-11) | `data_and_scripts/spikes_spontaneous/spiketimesmonkey2spont.mat` |

The analysis uses spike times from these recordings, not the associated
stimulus movies or every recording session in each collection. Download the
archive or files containing the listed recordings and retain their directory
structure. For PVC-3, keep all area-17 `t*.spk` files: the code performs its own
initial node screening before selecting ten nodes. Do not preselect a subset
of ten input files by hand. Retain accompanying metadata and documentation,
including `spk_info.inf` when provided.

## File placement

By default, `scripts/run_realdata_experiments.py` searches below `data/raw`
relative to the project root.

| Reader | Recognized path below the raw-data root |
| --- | --- |
| ret-1 | `crcns_ret-1/Data/20080516_R1.mat`; otherwise a recursive search for that exact filename |
| PVC-3 | `**/spont_activity/spike_data_area17/t*.spk` |
| PVC-11 | `**/data_and_scripts/spikes_spontaneous/spiketimesmonkey2spont.mat` |

`**` denotes any enclosing dataset directories, not a literal directory name.
Keep the displayed directory suffixes unchanged. Avoid storing multiple copies
of the same recording beneath one raw-data root. The ret-1 reader prefers the
canonical path shown above, then uses the first sorted recursive match if that
path is absent. The other two readers use the first sorted matching path.

If the recordings are already stored elsewhere, supply that location without
copying them into the code directory:

```text
python scripts/run_realdata_experiments.py --raw-root "PATH_TO_RAW_DATA" --windows 1
```

Use the same `--raw-root` on later commands in the workflow described in
`README.md`. A full rerun reads the fixed dataset choices, window settings,
and seed rules from `configs/realdata_experiments.json`; no manual renaming of
neurons or changes to those scientific settings are required.

## Citations and result archives

Use the dataset citations and associated publication citations requested on
each official dataset page. The records distributed here contain fitted
parameters, convergence diagnostics, node mappings, window definitions,
configurations, and seeds. They do not contain the underlying spike trains.

The neural runner's local results ZIP includes extracted event arrays
(`events.npz`). These arrays and other copies of the source recordings are
excluded from this publication archive. Obtain the recordings through the
official links above and follow their terms when sharing working results.
