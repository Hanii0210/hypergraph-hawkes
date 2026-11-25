# Hyperedge-triggered Hawkes processes

Research code and saved study records for **Hyperedge-Triggered Hawkes Processes
for Diagnosing Higher-Order Temporal Dependence in Event Streams**, by Zihan Xu,
*Journal of Statistical Computation and Simulation*.
Article DOI: [10.1080/00949655.2026.2762892](https://doi.org/10.1080/00949655.2026.2762892).

The archive contains the numerical implementation, fixed study configurations,
and figure/table inputs used for the reported results, together with compact
run records and a record-verification script. The numerical implementation,
configuration values, seed rules, and saved paper inputs are unchanged.
The studies use free group coefficients; the tensor module is retained as a
dependency of the shared inference code.

Extract the archive into a new directory and run commands from its
`hypergraph_hawkes` root. The three workflows below serve different purposes:
rebuild the paper outputs, inspect the saved records, or rerun the experiments.

## Environment

Install the dependencies in a Python environment:

```text
python -m pip install -r requirements.txt
```

The study run metadata record Python 3.14.7, NumPy 2.5.2, and SciPy 1.18.0 on
Windows 11. `requirements.txt` specifies lower bounds rather than a locked
environment. Numerical endpoints and timings can depend on the platform.
NumPy 2.0 or later is required for `numpy.trapezoid`.

## 1. Rebuild figures and tables

```text
python paper/make_tables.py
python paper/make_figures.py
```

These commands use the saved full-precision inputs in `paper/data`; they do not
simulate or fit models and need no raw neural recordings. They write to
`paper/tables`, `paper/figure`, and the generated CSV files in `paper/data`.
The plotting scripts request Times New Roman and STIX mathematics. Install
Times New Roman if the original typography is required.

## 2. Verify the saved records

```text
python scripts/verify_saved_records.py
```

The verifier checks the archived record counts, seeds and start selection,
recorded convergence diagnostics, correspondence with the paper inputs, and
benchmark summaries. It reads `records` and `paper/data`; it does not load raw
spike trains, simulate events, or refit models. Its default output is a JSON
report on the console. A successful check exits with status 0.

To check the release file hashes as well, run this command before regenerating
figures or tables, which can overwrite files covered by the checksum list:

```text
python scripts/verify_saved_records.py --verify-checksums
```

The compact start records retain endpoint diagnostics and summaries of the
optimization histories, not every iteration. Verification of these records is
distinct from recalculating likelihoods and gradients from the original event
streams. See [records and schema](docs/run_records.md) for the contents and
scope of this archive.

## 3. Rerun the experiments

A full rerun creates new working results under `experiments/results`. The
compact records in `records` are for inspection, not a resumable working tree.
The fixed configurations and runner behavior are described in
[reproduction scope](docs/reproduction_scope.md).

### Synthetic studies

Inspect the plan, then run the pilot and formal studies:

```text
python scripts/run_synthetic_experiments.py --phase formal --plan-only
python scripts/run_synthetic_experiments.py --phase pilot --repetitions 10
python scripts/run_synthetic_experiments.py --phase formal
```

The formal design has 1100 distinct paths, 4200 model fits, and three starts per
fit. The constant-background controls reuse specified paths. Pilot results do
not enter formal summaries. The recorded formal run took approximately
10 hours 20 minutes on the study machine. See
[synthetic design](docs/synthetic_experiments.md) for the seed scheme and outputs.

### Neural analysis

Download the required CRCNS recordings and place them below `data/raw` as
described in [data access](docs/data_access.md). The raw recordings and extracted
spike arrays are not distributed in this archive.

```text
python scripts/run_realdata_experiments.py --windows 5 --plan-only
python scripts/run_realdata_experiments.py --windows 1
python scripts/run_realdata_experiments.py --windows 5
python scripts/run_realdata_experiments.py --windows 5 --continue-unconverged
```

Use `--raw-root PATH` on each execution command if the recordings are elsewhere.
The first-window stage is reused by the five-window analysis. Continuation
extends saved unqualified coordinate endpoints to at most 2000 cumulative
sweeps and retains the training-based qualification rule. Inspect the final
`summary.json`, including after continuation.

The runner supports continuation at the five-window stage. If a newly run
first-window fit is unqualified, the readiness check can block that stage;
inspect its saved diagnostics before proceeding. See
[neural design](docs/realdata_experiments.md).

### Computational benchmark

```text
python experiments/syn07_scalability.py --plan-only
python experiments/syn07_scalability.py
```

The benchmark has nine fixed tasks, three measured phases, and ten timing
repetitions per phase. The 270 measurements are repeated timings, not 270
independent simulated paths. See [benchmark design](docs/scalability.md).

### Export a new run

After all studies finish, export their working results to a separate folder:

```text
python scripts/export_paper_results.py --output-dir reproduced_paper
python reproduced_paper/make_tables.py
python reproduced_paper/make_figures.py
```

The exporter reads the result trees under `experiments/results`. Its
`--synthetic`, `--realdata`, and `--timing` options also accept the corresponding
study-generated ZIP files. The compact `records` tree is not an exporter input.
The exporter does not fit models; it exports numerical inputs for the figures
and tables, not revised manuscript prose or captions.

Study runners also create `hypergraph_hawkes_results.zip` in Downloads and
preserve an earlier ZIP under `Downloads/archive/hypergraph_hawkes`. Use
`--download-dir PATH` for synthetic/neural runs or `--output-dir PATH` for the
benchmark to change this location. Interrupted runs reuse saved working
records. Inspect planned counts and qualification fields in `summary.json`;
the top-level invocation status alone does not certify every fit.

## Code and record map

| Component | Location |
| --- | --- |
| Completion features and exposures | `models/features.py`, `models/kernel.py`, `models/likelihood.py` |
| Constant and varying backgrounds | `models/baseline.py` |
| Source allocation and coefficient updates | `inference/e_step.py`, `inference/m_step.py`, `inference/em.py` |
| Coordinate maximization and KKT checks | `inference/coordinate.py` |
| Event generation | `simulation/simulator.py` |
| Synthetic study and seed rules | `experiments/synthetic_pipeline.py` |
| Chronological neural prediction | `experiments/realdata_baseline.py` |
| Timing study | `experiments/syn07_scalability.py` |
| Figure/table inputs | `paper/data` |
| Saved endpoint records and start summaries | `records`; [schema](docs/run_records.md) |

Node labels in Python are zero-based; manuscript group labels are one-based.
Ordinary coefficients use source-by-target orientation. Small implementation
checks can be run with `python scripts/run_model_checks.py`; these are not a
rerun of the full study.

Completion-window membership uses the floating-point arithmetic of the
reported runs. Exact-boundary sensitivity is documented in
[reproduction scope](docs/reproduction_scope.md).
`CITATION.cff` contains the article citation. `FILELIST.sha256` lists release
files and their checksums.
