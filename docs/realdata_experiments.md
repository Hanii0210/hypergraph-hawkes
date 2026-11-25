# Neural design

`configs/realdata_experiments.json` specifies three recordings, five later
windows per recording, initial node screening, top-one/top-two/top-three
candidate prefixes, constant and linear backgrounds, all ordinary
coefficients, and zero penalty. The main comparison uses the top three
candidates. Beta is 2 per second and Delta is 0.05 seconds; these are common
analysis scales, not estimated biological time scales.

`make_plan` in `experiments/realdata_baseline.py` records seeds for every
recording, window, model, and start. Candidate ranking uses the first third of
each window, coefficient estimation uses the middle third, and prediction
uses the last third. Paired comparisons share the relevant past events and
candidate prefixes. Tied timestamps use a common strict past followed by a
batch update.

The first-window stage is reused within the five-window study. The full plan
has 15 windows, 135 models including 15 screening fits, and three starts per
model. The implementation can reuse a fit when actual candidate sets are
identical; the recorded formal study contains 135 distinct executed fits.

The initial coordinate budget is 200 sweeps. `--continue-unconverged` extends
saved unqualified endpoints to at most 2000 cumulative sweeps. Continuation
is selected by training qualification, not test performance. In the recorded
study, nine C-P starts in ret-1 windows 03, 04, and 05 were continued; candidate
prefixes, seeds, and stopping tolerances were unchanged. All 405 final starts
and all 135 selected fits qualified.

The runner exposes continuation at the five-window stage. An unqualified
first-window run can block its readiness check; inspect the saved diagnostics
before proceeding with a new analysis.

Publication metadata and candidate records are in `records/realdata`.
`paper/data/realdata_fits.json` contains the 15 complete window results and
their 135 selected models. Current and pre-continuation start summaries are
separate files in `records/derived`; see [record schema](run_records.md).
Raw recordings are obtained as described in [data access](data_access.md).

New runs write to `experiments/results/realdata/baseline`, including the full
status in `realdata_experiments/full/summary.json`. Positive and negative
prediction differences are retained. Windows from one recording are dependent
descriptive observations.
