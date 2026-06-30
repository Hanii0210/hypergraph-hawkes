# Hyperedge-triggered Hawkes Processes

Statistical inference of higher-order interaction structure from asynchronous event streams.

This repository implements the Hyperedge-triggered Hawkes (HTH) process, a point-process model in which each node intensity contains ordinary pairwise Hawkes excitation plus an additional term activated when all members of a candidate hyperedge fire within a short temporal window.

The current project version focuses on three goals:

1. a closed-form EM implementation for pairwise and hyperedge-triggered terms;
2. statistically validated simulation and synthetic stress tests;
3. honest real-data model comparison using held-out candidate-count BIC.

## Model

For node \(n\),

\[
\lambda_n(t)
=
\mu_n
+
\sum_{j:t_j<t}\alpha_{n_j\to n}\phi(t-t_j)
+
\sum_{e\ni n}\alpha_e\phi(t-t_{\mathrm{anchor}}(e,t)).
\]

The hyperedge anchor is the most recent pattern-completion time: all members of candidate hyperedge \(e\) must have fired within a completion window \(\Delta\). The implementation uses a piecewise compensator so that each anchor is integrated only until the next completion event.

## Current formal real-data rule

For real data, the primary statistic is the held-out candidate-count BIC difference:

\[
\mathrm{BICdiff}
=
2(\log L_{\mathrm{HTH}}-\log L_{\mathrm{pairwise}})
-
|\mathcal{E}_{\mathrm{cand}}|\log(n_{\mathrm{heldout}}).
\]

Active-edge-count BIC is diagnostic only. If L1 shrinkage drives hyperedge coefficients to approximately zero, or if no active hyperedges survive, then likelihood gain alone is not treated as decisive HTH evidence.

## Repository structure

```text
hypergraph_hawkes/
|-- models/
|   |-- kernel.py
|   |-- likelihood.py
|   `-- tensor_param.py
|-- inference/
|   |-- e_step.py
|   |-- m_step.py
|   |-- em.py
|   `-- candidate_filter.py
|-- simulation/
|   |-- simulator.py
|   `-- data_loader.py
|-- experiments/
|   |-- exp*.py  # main and supplementary synthetic scripts
|   |-- realdata_pipeline.py
|   |-- realdata_ret1.py
|   |-- realdata_pvc3.py
|   |-- realdata_pvc11.py
|   |-- plot_realdata_summary.py
|   |-- checks/
|   |   `-- smoke_hth_bic_checked.py
|   |-- schematics/
|   |   |-- make_hth_activation_landscape.py
|   |   `-- plot_hth_model_schematic.py
|   |-- results/
|   `-- EXPERIMENTS.md
|-- tests/
|   |-- test_kernel.py
|   |-- test_tensor.py
|   |-- test_estep.py
|   |-- test_mstep.py
|   |-- test_simulator.py
|   |-- test_simulator_validity.py
|   |-- test_candidate_filter.py
|   `-- test_data_loader.py
|-- figures/
|-- archive/
|-- run_tests.py
|-- run_all.py
`-- README.md
```

## Installation

```powershell
pip install -r requirements.txt
```

The code is developed for Python 3.10+.

## Test suite

Run:

```powershell
python run_tests.py
```

Expected status:

```text
8 / 8 test files passed
```

The full test suite includes `test_simulator_validity.py`, a time-rescaling validity gate for the simulator. It can take about 1-2 minutes and may appear slower than the ordinary unit tests.

To run the simulator validity gate alone:

```powershell
python tests\test_simulator_validity.py
```

Expected output:

```text
Both time-rescaling gates PASS.
```

## Full pipeline

Run the full selected pipeline:

```powershell
python run_all.py
```

Useful alternatives:

```powershell
python run_all.py --skip-realdata
python run_all.py --skip-synthetic
python run_all.py --quick
python run_all.py --continue-on-error
```

`--quick` runs the test suite and regenerates cached real-data summary figures. It does not rerun the long synthetic suite or the raw real-data scripts.

The optional 3D schematic is not run by default:

```powershell
python run_all.py --include-schematic
```

## Synthetic experiments

The synthetic suite is currently organized as:

For the current experiment numbering, active/supplement status, and legacy-script notes, see `experiments/EXPERIMENTS.md`.

| Paper ID | Script | Purpose |
|---:|---|---|
| Syn 1 | `experiments/syn01_recovery_robustness.py` | parameter recovery and robustness |
| Syn 2 | `experiments/syn02_regularization_path.py` | regularization path and sparsity control |
| Syn 3 | `experiments/syn03_em_convergence.py` | EM convergence from random initializations |
| Syn 4 | `experiments/syn04_strength_sensitivity.py` | interaction-strength sensitivity and non-explosive behavior |
| Syn 5 | `experiments/syn05_likelihood_separation.py` | pairwise confounding and likelihood separation |
| Syn 6 | `experiments/syn06_trigger_window_sensitivity.py` | trigger-window sensitivity |
| Syn 7 | `experiments/syn07_scalability.py` | computational scalability |
| Syn 8 | `experiments/syn08_bias_ablation.py` | kernel-timescale bias/variance ablation |
| Syn 9 | `experiments/syn09_identification_diagnostic.py` | candidate nomination and detectability |
| Syn 10 | `experiments/syn10_interaction_baseline.py` | interaction-baseline comparison |

Supplementary scripts are kept for reproducibility but are not run by default:

| Supplement ID | Script | Purpose |
|---:|---|---|
| Supp A | `experiments/suppA_recovery_demo.py` | single-seed recovery demo |
| Supp B | `experiments/suppB_copula_validation.py` | copula tail-dependence validation |
| Supp C | `experiments/suppC_3node_hyperedge.py` | minimal 3-node hyperedge toy example |
| Supp D | `experiments/suppD_rank_sweep.py` | CP-rank sweep |
| Supp E | `experiments/suppE_calibration.py` | calibration / selective-inference diagnostic |

Formal real-data experiments are reported separately as R1--R3 below.

The old `archive/legacy_realdata_scripts/exp10_realdata.py` is legacy and is not used by the current formal real-data pipeline.

## Formal real-data pipeline

The formal real-data scripts use the shared implementation in:

```text
experiments/realdata_pipeline.py
```

Run all three formal datasets manually:

```powershell
python experiments\realdata_ret1.py --raw-root data\raw --file 20080516_R1.mat --record-index 0 --top-m-list 1,2,3 --n-iter 20

python experiments\realdata_pvc3.py --raw-root data\raw --top-m-list 1,2,3 --n-iter 20

python experiments\realdata_pvc11.py --raw-root data\raw --monkey 2 --top-m-list 1,2,3 --n-iter 20
```

Then generate summary tables and figures:

```powershell
python experiments\plot_realdata_summary.py
```

Main outputs:

```text
experiments/results/realdata/realdata_ret1_20080516_R1_rec0.csv
experiments/results/realdata/realdata_pvc3_area17.csv
experiments/results/realdata/realdata_pvc11_monkey2.csv
experiments/results/realdata/realdata_combined.csv
experiments/results/realdata/realdata_summary_by_dataset_topm.csv

figures/realdata/realdata_bic_stability.png
figures/realdata/realdata_positive_window_rate.png
figures/realdata/realdata_exposure_diagnostic_full.png
figures/realdata/realdata_exposure_diagnostic_clipped.png
```

## Current real-data interpretation

Formal held-out candidate-count BIC results:

| Dataset | top_m=1 | top_m=2 | top_m=3 | Interpretation |
|---|---:|---:|---:|---|
| ret-1 | mean +8.322, 5/5 positive | mean +1.856, 3/5 positive | mean -2.565, 2/5 positive | sparse but fragile support |
| PVC-3 area17 | mean +21.391, 5/5 positive | mean +12.822, 5/5 positive | mean +6.486, 4/5 positive | strongest stable positive case |
| PVC-11 monkey2 | mean +14.479, 5/5 positive | mean +10.821, 5/5 positive | mean +13.814, 4/5 positive | robust positive cortex case |

A binned pseudo-event G-Node retina check is not treated as decisive by candidate-count BIC.

## BIC smoke checks

Use:

```powershell
python experiments\checks\smoke_hth_bic_checked.py `
  --csv archive\legacy_realdata_outputs\exp10_realdata_events.csv `
  --T 40.0 `
  --top-m-pairs 1 `
  --n-iter 1 `
  --out archive\dev_smoke_outputs\hth_bic_checked_smoke_exp10_T40.pkl
```

Use the actual observation horizon for `--T`; do not set `T` longer than the recorded event window unless the missing interval is intentionally part of the observation period. The example above uses the archived legacy event CSV with `T=40.0`, matching its event time span.

This script verifies the model-comparison accounting. Candidate-count BIC is the primary formal statistic; active-count BIC is printed only as a diagnostic.

## Loading custom event data

CSV input should contain columns:

```text
time,node
0.123,0
0.456,2
0.789,1
```

The observation horizon `T` must be passed explicitly:

```python
from simulation.data_loader import load_events_from_csv, summarise_events

events, n_nodes, T = load_events_from_csv("myevents.csv", T=100.0)
summarise_events(events, n_nodes, T)
```

Do not infer `T` from the final observed event time. Doing so truncates the compensator and changes the likelihood.

## Notes for development

Current completed status:

```text
[done] BIC accounting smoke check
[done] shared realdata_pipeline.py
[done] formal ret-1 / PVC-3 / PVC-11 scripts
[done] real-data summary figures
[done] tests, including simulator validity gate
[pending] final paper text and final synthetic table numbering
[pending] final cleanup of legacy file names
```

Legacy scripts and files are intentionally kept until the paper/project outputs are fully stable.
