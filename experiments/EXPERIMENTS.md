# Experiment Inventory

This file records the paper-facing experiment order and the corresponding legacy code filenames.  
Main synthetic script filenames follow the paper-facing `syn01`--`syn10` order; supplementary scripts use `suppA`--`suppE` filenames.

## Main synthetic experiments

| Paper ID | Legacy file | Output artifacts | Role |
|---:|---|---|---|
| Syn 1 | `syn01_recovery_robustness.py` | `syn01_recovery_robustness.pkl/png` | parameter recovery and robustness |
| Syn 2 | `syn02_regularization_path.py` | `syn02_regularization_path.pkl/png` | regularization path and sparsity control |
| Syn 3 | `syn03_em_convergence.py` | `syn03_final_logliks.npy`, `syn03_trajectories.npy`, `syn03_em_convergence.png` | EM convergence from random initializations |
| Syn 4 | `syn04_strength_sensitivity.py` | `syn04_strength_sensitivity.pkl/png` | interaction-strength sensitivity and non-explosive behavior |
| Syn 5 | `syn05_likelihood_separation.py` | `syn05_likelihood_separation.pkl/png` | pairwise confounding and likelihood separation |
| Syn 6 | `syn06_trigger_window_sensitivity.py` | `syn06_trigger_window_sensitivity.pkl/png` | trigger-window sensitivity |
| Syn 7 | `syn07_scalability.py` | `syn07_scalability.pkl/png` | computational scalability |
| Syn 8 | `syn08_bias_ablation.py` | `syn08_bias_ablation.pkl/png` | kernel-timescale bias/variance ablation |
| Syn 9 | `syn09_identification_diagnostic.py` | `syn09_identification_diagnostic.pkl/png` | candidate nomination and detectability |
| Syn 10 | `syn10_interaction_baseline.py` | `syn10_interaction_baseline.pkl/png` | interaction-baseline comparison |

## Supplementary / optional synthetic experiments

These scripts are kept for reproducibility but are not part of the default `run_all.py` pipeline.

| Supplement ID | Legacy file | Output artifacts | Role |
|---:|---|---|---|
| Supp A | `suppA_recovery_demo.py` | demo only | single-seed recovery demo |
| Supp B | `suppB_copula_validation.py`, `suppB_plot_copula_validation.py` | `suppB_copula_validation.pkl/png` | copula tail-dependence validation |
| Supp C | `suppC_3node_hyperedge.py`, `suppC_plot_3node_hyperedge.py` | `suppC_3node_hyperedge.pkl/png` | minimal 3-node hyperedge toy example |
| Supp D | `suppD_generate_rank_data.py`, `suppD_rank_sweep.py`, `suppD_plot_rank_sweep.py` | `suppD_data_R2.pkl`, `suppD_data_R3.pkl`, `suppD_rank_sweep.pkl/png` | CP-rank sweep |
| Supp E | `suppE_selective_inference.py`, `suppE_calibration.py`, `suppE_plot_calibration.py` | `suppE_calibration.pkl/png`, `suppE_control.pkl` | calibration / selective-inference diagnostic |

## Formal real-data experiments

| Paper ID | Script | Output prefix | Role |
|---:|---|---|---|
| R1 | `real01_ret1.py` | `results/realdata/real01_ret1_20080516_R1_rec0.*` | ret-1 retina |
| R2 | `real02_pvc3.py` | `results/realdata/real02_pvc3_area17.*` | PVC-3 cortex |
| R3 | `real03_pvc11.py` | `results/realdata/real03_pvc11_monkey2.*` | PVC-11 cortex |

## Legacy real-data scripts

| Old ID | Archived path | Status |
|---:|---|---|
| 10 | `archive/legacy_realdata_scripts/exp10_realdata.py`, `archive/legacy_realdata_scripts/exp10_plot.py` | legacy old real-data experiment; superseded by the formal real-data pipeline |
| 16 | `archive/legacy_realdata_scripts/exp16_pvc3_window_stability.py` | legacy PVC-3 exploratory real-data script; superseded by `real02_pvc3.py` |
| 17 | `archive/legacy_realdata_scripts/exp17_pvc11_window_stability.py` | legacy PVC-11 exploratory real-data script; superseded by `real03_pvc11.py` |
| 18 | `archive/legacy_realdata_scripts/exp18_ret1_window_stability.py` | legacy ret-1 exploratory real-data script; superseded by `real01_ret1.py` |

## Notes

- Main synthetic scripts have been renamed to `syn01`--`syn10`. Supplementary scripts have been renamed to `suppA`--`suppE`, while output artifacts keep their original `exp*` names for compatibility.
- Supplementary scripts remain in `experiments/` because synthetic data artifacts are stored in `experiments/results/synthetic/`, and synthetic figures are stored in `figures/synthetic/`.
- Do not use `archive/legacy_realdata_scripts/exp10_realdata.py` for formal real-data claims.
- Do not use the archived `exp16`, `exp17`, or `exp18` legacy scripts for final real-data claims.
- Legacy real-data output artifacts are archived in `archive/legacy_realdata_outputs/`.
- Development smoke-test output artifacts are archived in `archive/dev_smoke_outputs/`.
- Development helper scripts are archived in `archive/dev_helper_scripts/`.
- `smoke_hth_bic_checked.py` is kept as `experiments/checks/smoke_hth_bic_checked.py`.
- Optional schematic scripts are kept in `experiments/schematics/`.
