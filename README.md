# Hyperedge-triggered Hawkes processes

Zihan Xu (2026). *Hyperedge-Triggered Hawkes Processes for Diagnosing Higher-Order Temporal Dependence in Event Streams*. Journal of Statistical Computation and Simulation. [DOI: 10.1080/00949655.2026.2762892](https://doi.org/10.1080/00949655.2026.2762892)

Does recent activity across an entire group help predict the next events on its members? A group completes when each member has had an event within a fixed time window. In the model, its most recent completion contributes an exponentially decaying term to each member's intensity. The paper measures the predictive contribution of three-node groups by comparing held-out log likelihoods after accounting for two-node completions.

## Reproduce the figures and tables

Run these commands from the repository root. Both scripts read the saved results in [paper/data](paper/data).

```sh
python -m pip install -r requirements.txt
python paper/make_figures.py
python paper/make_tables.py
```

- **Figures.** `paper/figure/` contains four main-text figures and three supplementary figures in PDF and PNG. Use `python paper/make_figures.py --main` for the four main-text figures alone.
- **Tables.** `paper/tables/` contains nine LaTeX tables covering prediction, parameter recovery, neural recordings, and computation times. The corresponding CSV files are in `paper/data/`.

## Run the analyses

**Simulation study.** The formal study uses 1,100 simulated paths to examine overlapping three-node groups and time-varying backgrounds. It fits each model from three starting points. The formal run took about 10 hours and 20 minutes on the study machine. [Design and seed settings](docs/synthetic_experiments.md)

```sh
python scripts/run_synthetic_experiments.py --phase pilot --repetitions 10
python scripts/run_synthetic_experiments.py --phase formal
```

**Neural recordings.** The analysis evaluates held-out prediction in five windows from each of three CRCNS recordings. Download the files listed in the [data guide](docs/data_access.md) into `data/raw/`. Fit the first window, run the five-window analysis, then continue fits that require more iterations. [Models and window selection](docs/realdata_experiments.md)

```sh
python scripts/run_realdata_experiments.py --windows 1
python scripts/run_realdata_experiments.py --windows 5
python scripts/run_realdata_experiments.py --windows 5 --continue-unconverged
```

**Timing study.** The benchmark measures feature construction, one cached EM update, and a complete fit as the event or node count varies. [Benchmark design](docs/scalability.md)

```sh
python experiments/syn07_scalability.py
```

After running all three analyses, generate the figures and tables from the new results.

```sh
python scripts/export_paper_results.py --output-dir reproduced_paper
python reproduced_paper/make_figures.py
python reproduced_paper/make_tables.py
```

## Model code and saved fits

The [model code](models) implements group completions, exposures, and the event log likelihood. The [estimation routines](inference) fit the coefficients by source-allocation EM and coordinate maximization. The [simulator](simulation/simulator.py) generates event streams.

Individual fit records include the selected starting point, training and test scores, and convergence diagnostics. To compare these records with the paper summaries, run `python scripts/verify_saved_records.py`. [Record layout and field definitions](docs/run_records.md)
